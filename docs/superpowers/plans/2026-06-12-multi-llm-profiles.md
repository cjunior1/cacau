# Multi-LLM Profiles & Auto-Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded `ChatAnthropic` LLM with a named-profile system supporting Anthropic, OpenAI, Google Gemini, Ollama, and Groq, plus an `auto` mode that uses a classifier LLM to select the best profile for each prompt.

**Architecture:** LLM profiles are defined in `config/settings.yaml` with a `description` field. A provider factory (`agent/providers.py`) instantiates the correct LangChain chat class per profile. An auto-selector (`agent/selector.py`) calls a fast classifier LLM to pick the best profile from the descriptions. The harness builds the graph with the selected LLM on every `run()` call. A health checker (`agent/health.py`) tests each profile on demand via `dev-agent config check`.

**Tech Stack:** LangChain (langchain-anthropic, langchain-openai, langchain-google-genai, langchain-groq, langchain-community for Ollama), LangGraph, Pydantic v2, Typer, Rich, pytest.

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `pyproject.toml` | Add optional provider extras |
| Modify | `config/settings.yaml` | Add `profiles` + `llm_selector` sections |
| Modify | `src/dev_agent/config.py` | Add `LLMProfile`, `LLMSelectorConfig`, update `Settings` |
| **Create** | `src/dev_agent/agent/providers.py` | Factory: profile → `BaseChatModel` |
| **Create** | `src/dev_agent/agent/selector.py` | Auto-select best profile via classifier LLM |
| **Create** | `src/dev_agent/agent/health.py` | Health-check all profiles |
| Modify | `src/dev_agent/agent/graph.py` | Accept `BaseChatModel` instead of building it |
| Modify | `src/dev_agent/agent/harness.py` | Wire selector + providers; emit `profile_selected` event |
| Modify | `src/dev_agent/cli/main.py` | Add `--profile` flag; add `config check` subcommand |
| Modify | `src/dev_agent/cli/repl.py` | Render `[auto → name · model]` badge |
| **Create** | `tests/test_providers.py` | Unit tests for provider factory |
| **Create** | `tests/test_selector.py` | Unit tests for auto-selector |
| **Create** | `tests/test_health.py` | Unit tests for health checker |

---

## Task 1: Add optional provider dependencies to `pyproject.toml`

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Open and edit `pyproject.toml`**

Replace the `[project.optional-dependencies]` section (currently only has `dev`) with:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.6.0",
]
anthropic = ["langchain-anthropic>=0.3.0"]
openai    = ["langchain-openai>=0.2.0"]
google    = ["langchain-google-genai>=2.0.0"]
groq      = ["langchain-groq>=0.2.0"]
all-llms  = [
    "langchain-anthropic>=0.3.0",
    "langchain-openai>=0.2.0",
    "langchain-google-genai>=2.0.0",
    "langchain-groq>=0.2.0",
]
```

Note: Ollama uses `langchain-community`, already in `[project.dependencies]`.

- [ ] **Step 2: Reinstall with all extras**

```bash
pip install -e ".[all-llms,dev]" -q
```

Expected: installs without errors (some packages may already be present).

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add optional provider extras for openai, google, groq"
```

---

## Task 2: Add `LLMProfile` and `LLMSelectorConfig` to `config.py`

**Files:**
- Modify: `src/dev_agent/config.py`
- Test: `tests/test_config_profiles.py` (inline verification, no new test file needed)

- [ ] **Step 1: Write a failing test for `LLMProfile` parsing**

Create `tests/test_config_profiles.py`:

```python
"""Tests for LLMProfile config parsing."""
from dev_agent.config import LLMProfile, LLMSelectorConfig, Settings


def test_llm_profile_fields():
    p = LLMProfile(
        provider="anthropic",
        model="claude-sonnet-4-6",
        api_key_env="ANTHROPIC_API_KEY",
        temperature=0.1,
        streaming=True,
        description="Test profile",
    )
    assert p.provider == "anthropic"
    assert p.api_key_env == "ANTHROPIC_API_KEY"
    assert p.base_url is None


def test_llm_profile_ollama_base_url():
    p = LLMProfile(
        provider="ollama",
        model="qwen2.5-coder:7b",
        base_url="http://localhost:11434",
        description="Local model",
    )
    assert p.base_url == "http://localhost:11434"
    assert p.api_key_env is None


def test_settings_has_profiles():
    s = Settings(profiles={
        "fast": LLMProfile(provider="groq", model="llama-3.3-70b-versatile", description="Fast")
    })
    assert "fast" in s.profiles
    assert s.profiles["fast"].provider == "groq"


def test_settings_default_profile_is_auto():
    s = Settings()
    assert s.agent.profile == "auto"


def test_llm_selector_config():
    c = LLMSelectorConfig(profile="fast")
    assert c.profile == "fast"
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
pytest tests/test_config_profiles.py -v
```

Expected: `ImportError` — `LLMProfile` does not exist yet.

- [ ] **Step 3: Rewrite `src/dev_agent/config.py`**

Replace the entire file with:

```python
"""Configuration management with pydantic-settings and YAML support."""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


_sub_config = SettingsConfigDict(extra="ignore")


class LLMProfile(BaseModel):
    """A named LLM configuration profile."""
    model_config = SettingsConfigDict(extra="ignore")

    provider: str                        # anthropic | openai | google | ollama | groq
    model: str
    description: str = ""
    api_key_env: str | None = None       # env var name holding the API key
    base_url: str | None = None          # for ollama or custom endpoints
    temperature: float = 0.1
    streaming: bool = True


class LLMSelectorConfig(BaseModel):
    model_config = SettingsConfigDict(extra="ignore")
    profile: str = "fast"               # profile name to use as classifier


class AgentConfig(BaseSettings):
    model_config = _sub_config
    profile: str = "auto"               # profile name or "auto"
    max_iterations: int = 25
    recursion_limit: int = 50
    streaming: bool = True              # fallback default; profile.streaming takes precedence


class HarnessConfig(BaseSettings):
    model_config = _sub_config
    checkpointing: bool = True
    interrupt_before: list[str] = Field(default_factory=list)
    interrupt_after: list[str] = Field(default_factory=list)
    debug_mode: bool = False


class WebhookConfig(BaseSettings):
    model_config = _sub_config
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8080
    secret: str = ""


_DEFAULT_PROFILES: dict[str, LLMProfile] = {
    "default": LLMProfile(
        provider="anthropic",
        model="claude-sonnet-4-6",
        api_key_env="ANTHROPIC_API_KEY",
        description="Default Anthropic Claude profile for general development tasks.",
    )
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DEV_AGENT_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    profiles: dict[str, LLMProfile] = Field(default_factory=lambda: dict(_DEFAULT_PROFILES))
    llm_selector: LLMSelectorConfig = Field(default_factory=LLMSelectorConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    harness: HarnessConfig = Field(default_factory=HarnessConfig)
    webhooks: WebhookConfig = Field(default_factory=WebhookConfig)
    workspace_dir: str = Field(default=".", alias="DEV_AGENT_WORKSPACE")

    @classmethod
    def from_yaml(cls, path: Path) -> "Settings":
        if not path.exists():
            return cls()
        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        raw_profiles = data.pop("profiles", {})
        profiles = {name: LLMProfile(**cfg) for name, cfg in raw_profiles.items()} if raw_profiles else dict(_DEFAULT_PROFILES)

        raw_selector = data.pop("llm_selector", {})
        llm_selector = LLMSelectorConfig(**raw_selector) if raw_selector else LLMSelectorConfig()

        agent_data = data.pop("agent", {})
        harness_data = data.pop("harness", {})
        webhooks_data = data.pop("webhooks", {})
        data.pop("tools", None)
        data.pop("cli", None)

        return cls(
            profiles=profiles,
            llm_selector=llm_selector,
            agent=AgentConfig(**agent_data),
            harness=HarnessConfig(**harness_data),
            webhooks=WebhookConfig(**webhooks_data),
            **data,
        )

    def get_profile(self, name: str | None = None) -> LLMProfile:
        """Return the named profile, or the agent's configured profile."""
        key = name or self.agent.profile
        if key == "auto" or key not in self.profiles:
            # fallback to first defined profile
            return next(iter(self.profiles.values()))
        return self.profiles[key]


_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_yaml(_CONFIG_PATH)
    return _settings


def reset_settings() -> None:
    """Force reload on next get_settings() call. Used in tests."""
    global _settings
    _settings = None
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_config_profiles.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Confirm existing tools tests still pass**

```bash
pytest tests/test_tools.py -v
```

Expected: 11 passed.

- [ ] **Step 6: Commit**

```bash
git add src/dev_agent/config.py tests/test_config_profiles.py
git commit -m "feat: add LLMProfile and LLMSelectorConfig to config"
```

---

## Task 3: Update `config/settings.yaml` with profiles

**Files:**
- Modify: `config/settings.yaml`

- [ ] **Step 1: Replace `config/settings.yaml`**

```yaml
profiles:
  powerful:
    provider: anthropic
    model: claude-opus-4-8
    api_key_env: ANTHROPIC_API_KEY
    temperature: 0.1
    streaming: true
    description: >
      Best for complex reasoning, architecture decisions, and reviewing
      large codebases. Use when the task requires deep analysis or multi-step planning.

  balanced:
    provider: anthropic
    model: claude-sonnet-4-6
    api_key_env: ANTHROPIC_API_KEY
    temperature: 0.1
    streaming: true
    description: >
      Good balance of speed and quality. Suitable for debugging, writing
      unit tests, medium-complexity refactors, and code explanations.

  fast:
    provider: groq
    model: llama-3.3-70b-versatile
    api_key_env: GROQ_API_KEY
    temperature: 0.2
    streaming: true
    description: >
      Fast and cheap. Best for simple tasks: formatting code, explaining
      short functions, quick Q&A, and anything that does not require deep reasoning.

  openai:
    provider: openai
    model: gpt-4o-mini
    api_key_env: OPENAI_API_KEY
    temperature: 0.1
    streaming: true
    description: >
      OpenAI GPT-4o-mini. Good alternative for debugging and refactoring
      when Anthropic is unavailable.

  local:
    provider: ollama
    model: qwen2.5-coder:7b
    base_url: http://localhost:11434
    description: >
      Local model, no API cost, no data leaves the machine. Use for
      sensitive codebases or offline environments.

agent:
  profile: auto
  max_iterations: 25
  recursion_limit: 50
  streaming: true

llm_selector:
  profile: fast

harness:
  checkpointing: true
  interrupt_before: []
  interrupt_after: []
  debug_mode: false

tools:
  enabled:
    - shell
    - file_read
    - file_write
    - git
    - code_search
    - web_fetch
    - code_lint
    - test_runner

webhooks:
  enabled: false
  host: "0.0.0.0"
  port: 8080
  secret: ""

cli:
  history_file: "~/.dev_agent_history"
  max_history: 500
  theme: "monokai"
```

- [ ] **Step 2: Verify settings load correctly**

```bash
python -c "
from dev_agent.config import get_settings, reset_settings
reset_settings()
s = get_settings()
print('profiles:', list(s.profiles.keys()))
print('agent.profile:', s.agent.profile)
print('selector.profile:', s.llm_selector.profile)
"
```

Expected output:
```
profiles: ['powerful', 'balanced', 'fast', 'openai', 'local']
agent.profile: auto
selector.profile: fast
```

- [ ] **Step 3: Commit**

```bash
git add config/settings.yaml
git commit -m "config: add multi-provider LLM profiles to settings.yaml"
```

---

## Task 4: Create `agent/providers.py` — provider factory

**Files:**
- Create: `src/dev_agent/agent/providers.py`
- Test: `tests/test_providers.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_providers.py`:

```python
"""Tests for the LLM provider factory."""
from unittest.mock import MagicMock, patch

import pytest

from dev_agent.config import LLMProfile
from dev_agent.agent.providers import build_llm, ConfigError


def _profile(**kwargs) -> LLMProfile:
    defaults = dict(provider="anthropic", model="claude-sonnet-4-6",
                    api_key_env="ANTHROPIC_API_KEY", description="test")
    return LLMProfile(**{**defaults, **kwargs})


def test_unknown_provider_raises():
    p = _profile(provider="unknown_provider")
    with pytest.raises(ConfigError, match="Unknown provider"):
        build_llm(p, tools=[])


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    p = _profile(provider="anthropic", api_key_env="ANTHROPIC_API_KEY")
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        build_llm(p, tools=[])


def test_missing_package_raises(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    p = _profile(provider="openai", api_key_env="OPENAI_API_KEY", model="gpt-4o-mini")
    with patch.dict("sys.modules", {"langchain_openai": None}):
        with pytest.raises(ConfigError, match="pip install dev-agent\[openai\]"):
            build_llm(p, tools=[])


def test_anthropic_build(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    p = _profile(provider="anthropic")
    mock_cls = MagicMock()
    mock_cls.return_value.bind_tools.return_value = MagicMock()
    with patch("dev_agent.agent.providers._import_anthropic", return_value=mock_cls):
        llm = build_llm(p, tools=[])
    mock_cls.assert_called_once_with(
        model="claude-sonnet-4-6", temperature=0.1,
        api_key="sk-ant-test", streaming=True,
    )


def test_ollama_skips_api_key_check():
    p = LLMProfile(provider="ollama", model="qwen2.5-coder:7b",
                   base_url="http://localhost:11434", description="local")
    mock_cls = MagicMock()
    mock_cls.return_value.bind_tools.return_value = MagicMock()
    with patch("dev_agent.agent.providers._import_ollama", return_value=mock_cls):
        llm = build_llm(p, tools=[])
    mock_cls.assert_called_once_with(
        model="qwen2.5-coder:7b", base_url="http://localhost:11434"
    )
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_providers.py -v
```

Expected: `ImportError` — `providers` module does not exist.

- [ ] **Step 3: Create `src/dev_agent/agent/providers.py`**

```python
"""Provider factory — instantiates the correct BaseChatModel for a given LLMProfile."""

import os
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

from dev_agent.config import LLMProfile

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


class ConfigError(Exception):
    """Raised when a profile is misconfigured (missing key, missing package, etc.)."""


def _resolve_api_key(profile: LLMProfile) -> str:
    if not profile.api_key_env:
        return ""
    key = os.environ.get(profile.api_key_env, "")
    if not key:
        raise ConfigError(
            f"{profile.api_key_env} environment variable is not set. "
            f"Add it to your .env file or export it in your shell."
        )
    return key


def _import_anthropic():
    try:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic
    except ImportError:
        raise ConfigError("langchain-anthropic is not installed. Run: pip install dev-agent[anthropic]")


def _import_openai():
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI
    except ImportError:
        raise ConfigError("langchain-openai is not installed. Run: pip install dev-agent[openai]")


def _import_google():
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI
    except ImportError:
        raise ConfigError(
            "langchain-google-genai is not installed. Run: pip install dev-agent[google]"
        )


def _import_groq():
    try:
        from langchain_groq import ChatGroq
        return ChatGroq
    except ImportError:
        raise ConfigError("langchain-groq is not installed. Run: pip install dev-agent[groq]")


def _import_ollama():
    try:
        from langchain_community.chat_models import ChatOllama
        return ChatOllama
    except ImportError:
        raise ConfigError(
            "langchain-community is not installed. Run: pip install langchain-community"
        )


def build_llm(profile: LLMProfile, tools: list[BaseTool]) -> "BaseChatModel":
    """Instantiate the correct chat model for the given profile and bind tools."""
    provider = profile.provider.lower()

    if provider == "anthropic":
        api_key = _resolve_api_key(profile)
        cls = _import_anthropic()
        llm = cls(
            model=profile.model,
            temperature=profile.temperature,
            api_key=api_key,
            streaming=profile.streaming,
        )

    elif provider == "openai":
        api_key = _resolve_api_key(profile)
        cls = _import_openai()
        llm = cls(
            model=profile.model,
            temperature=profile.temperature,
            api_key=api_key,
            streaming=profile.streaming,
        )

    elif provider == "google":
        api_key = _resolve_api_key(profile)
        cls = _import_google()
        llm = cls(
            model=profile.model,
            temperature=profile.temperature,
            google_api_key=api_key,
        )

    elif provider == "groq":
        api_key = _resolve_api_key(profile)
        cls = _import_groq()
        llm = cls(
            model=profile.model,
            temperature=profile.temperature,
            groq_api_key=api_key,
            streaming=profile.streaming,
        )

    elif provider == "ollama":
        cls = _import_ollama()
        kwargs: dict = {"model": profile.model}
        if profile.base_url:
            kwargs["base_url"] = profile.base_url
        llm = cls(**kwargs)

    else:
        raise ConfigError(
            f"Unknown provider '{profile.provider}'. "
            f"Supported: anthropic, openai, google, groq, ollama."
        )

    return llm.bind_tools(tools)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_providers.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/dev_agent/agent/providers.py tests/test_providers.py
git commit -m "feat: add provider factory supporting anthropic, openai, google, groq, ollama"
```

---

## Task 5: Create `agent/selector.py` — auto profile selector

**Files:**
- Create: `src/dev_agent/agent/selector.py`
- Test: `tests/test_selector.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_selector.py`:

```python
"""Tests for the LLM auto-selector."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dev_agent.config import LLMProfile
from dev_agent.agent.selector import select_profile


PROFILES = {
    "powerful": LLMProfile(provider="anthropic", model="claude-opus-4-8",
                            description="Best for complex reasoning and architecture."),
    "fast": LLMProfile(provider="groq", model="llama-3.3-70b-versatile",
                       description="Fast and cheap for simple tasks."),
}


@pytest.mark.asyncio
async def test_select_returns_valid_profile():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="powerful"))

    result = await select_profile("refactor the payments module", PROFILES, mock_llm)
    assert result == "powerful"


@pytest.mark.asyncio
async def test_select_strips_whitespace_and_quotes():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content='  "fast"  '))

    result = await select_profile("explain this function quickly", PROFILES, mock_llm)
    assert result == "fast"


@pytest.mark.asyncio
async def test_select_falls_back_on_unknown_profile():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="nonexistent_profile"))

    result = await select_profile("do something", PROFILES, mock_llm)
    assert result == "powerful"  # first profile in dict


@pytest.mark.asyncio
async def test_select_falls_back_on_exception():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("network error"))

    result = await select_profile("do something", PROFILES, mock_llm)
    assert result == "powerful"  # first profile
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_selector.py -v
```

Expected: `ImportError` — module does not exist.

- [ ] **Step 3: Create `src/dev_agent/agent/selector.py`**

```python
"""Auto-selector: uses a classifier LLM to pick the best profile for a prompt."""

import logging
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage

from dev_agent.config import LLMProfile

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

log = logging.getLogger(__name__)

_CLASSIFIER_PROMPT = """\
You are an LLM router. Given the user's task and the available LLM profiles below,
reply with ONLY the name of the most suitable profile — nothing else, no explanation.

Available profiles:
{profiles}

User task: {prompt}"""


async def select_profile(
    prompt: str,
    profiles: dict[str, LLMProfile],
    classifier_llm: "BaseChatModel",
) -> str:
    """Call the classifier LLM and return the selected profile name.

    Falls back to the first profile in `profiles` if the classifier fails
    or returns an unknown name.
    """
    fallback = next(iter(profiles))
    profile_lines = "\n".join(
        f"  {name}: {p.description.strip()}" for name, p in profiles.items()
    )
    classifier_input = _CLASSIFIER_PROMPT.format(profiles=profile_lines, prompt=prompt)

    try:
        response = await classifier_llm.ainvoke([HumanMessage(content=classifier_input)])
        chosen = response.content.strip().strip('"').strip("'").strip()
        if chosen not in profiles:
            log.warning("Classifier returned unknown profile '%s', falling back to '%s'", chosen, fallback)
            return fallback
        log.info("Auto-selected profile: %s", chosen)
        return chosen
    except Exception as exc:
        log.error("Profile selector failed (%s), falling back to '%s'", exc, fallback)
        return fallback
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_selector.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/dev_agent/agent/selector.py tests/test_selector.py
git commit -m "feat: add LLM auto-selector using classifier LLM"
```

---

## Task 6: Create `agent/health.py` — profile health checker

**Files:**
- Create: `src/dev_agent/agent/health.py`
- Test: `tests/test_health.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_health.py`:

```python
"""Tests for profile health checker."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dev_agent.config import LLMProfile
from dev_agent.agent.health import check_profile, check_all, ProfileStatus


PROFILES = {
    "fast": LLMProfile(provider="groq", model="llama-3.3-70b-versatile",
                       api_key_env="GROQ_API_KEY", description="Fast"),
    "local": LLMProfile(provider="ollama", model="qwen2.5-coder:7b",
                        base_url="http://localhost:11434", description="Local"),
}


@pytest.mark.asyncio
async def test_check_profile_ok(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Paris"))

    with patch("dev_agent.agent.health.build_llm", return_value=mock_llm):
        status = await check_profile("fast", PROFILES["fast"])

    assert status.name == "fast"
    assert status.ok is True
    assert "Paris" in status.snippet
    assert status.error is None


@pytest.mark.asyncio
async def test_check_profile_missing_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    status = await check_profile("fast", PROFILES["fast"])

    assert status.ok is False
    assert "GROQ_API_KEY" in status.error


@pytest.mark.asyncio
async def test_check_profile_llm_error(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("connection refused"))

    with patch("dev_agent.agent.health.build_llm", return_value=mock_llm):
        status = await check_profile("fast", PROFILES["fast"])

    assert status.ok is False
    assert "connection refused" in status.error


@pytest.mark.asyncio
async def test_check_all_returns_all_profiles():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Paris"))

    with patch("dev_agent.agent.health.build_llm", return_value=mock_llm):
        results = await check_all(PROFILES)

    assert len(results) == 2
    names = [r.name for r in results]
    assert "fast" in names
    assert "local" in names
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_health.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Create `src/dev_agent/agent/health.py`**

```python
"""Profile health checker — tests each LLM profile with a simple call."""

import asyncio
import time
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage

from dev_agent.agent.providers import build_llm, ConfigError
from dev_agent.config import LLMProfile

_PING_PROMPT = "What is the capital of France? Reply in one word."


@dataclass
class ProfileStatus:
    name: str
    provider: str
    model: str
    ok: bool
    latency_ms: float = 0.0
    snippet: str = ""
    error: str | None = None


async def check_profile(name: str, profile: LLMProfile) -> ProfileStatus:
    """Call the profile with a simple ping and return its status."""
    base = ProfileStatus(name=name, provider=profile.provider, model=profile.model, ok=False)
    try:
        llm = build_llm(profile, tools=[])
    except ConfigError as e:
        base.error = str(e)
        return base

    t0 = time.monotonic()
    try:
        response = await llm.ainvoke([HumanMessage(content=_PING_PROMPT)])
        elapsed = (time.monotonic() - t0) * 1000
        snippet = (getattr(response, "content", "") or "")[:60].strip()
        return ProfileStatus(
            name=name, provider=profile.provider, model=profile.model,
            ok=True, latency_ms=round(elapsed, 1), snippet=snippet,
        )
    except Exception as exc:
        elapsed = (time.monotonic() - t0) * 1000
        base.latency_ms = round(elapsed, 1)
        base.error = str(exc)
        return base


async def check_all(profiles: dict[str, LLMProfile]) -> list[ProfileStatus]:
    """Check all profiles concurrently and return their statuses."""
    tasks = [check_profile(name, profile) for name, profile in profiles.items()]
    return await asyncio.gather(*tasks)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_health.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/dev_agent/agent/health.py tests/test_health.py
git commit -m "feat: add profile health checker"
```

---

## Task 7: Update `agent/graph.py` — accept `BaseChatModel` directly

**Files:**
- Modify: `src/dev_agent/agent/graph.py`

- [ ] **Step 1: Replace `src/dev_agent/agent/graph.py`**

```python
"""LangGraph Deep Agent graph construction."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from dev_agent.agent.state import AgentState
from dev_agent.agent.prompts import build_system_prompt
from dev_agent.config import Settings


def build_graph(llm: BaseChatModel, settings: Settings):
    """Build and compile the deep agent StateGraph.

    Accepts a BaseChatModel already bound to tools (via providers.build_llm).
    Implements a ReAct loop with a max-iterations guard.
    """
    max_iter = settings.agent.max_iterations

    def agent_node(state: AgentState) -> dict:
        system_msg = SystemMessage(content=build_system_prompt(state.get("workspace", ".")))
        messages = [system_msg] + list(state["messages"])
        response = llm.invoke(messages)
        count = state.get("tool_calls_count", 0)
        if hasattr(response, "tool_calls") and response.tool_calls:
            count += len(response.tool_calls)
        return {"messages": [response], "tool_calls_count": count}

    def should_continue(state: AgentState) -> str:
        if state.get("tool_calls_count", 0) >= max_iter:
            return END
        return tools_condition(state)

    # ToolNode requires a list of tools; extract them from the bound LLM's kwargs
    bound_tools = getattr(llm, "tools", None) or []
    tool_node = ToolNode(bound_tools)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph
```

- [ ] **Step 2: Verify existing tests still pass**

```bash
pytest tests/ -v
```

Expected: all previous tests pass (graph is not directly tested, harness will be updated next).

- [ ] **Step 3: Commit**

```bash
git add src/dev_agent/agent/graph.py
git commit -m "refactor: graph.py accepts BaseChatModel instead of building it internally"
```

---

## Task 8: Update `agent/harness.py` — wire selector and providers

**Files:**
- Modify: `src/dev_agent/agent/harness.py`

- [ ] **Step 1: Replace `src/dev_agent/agent/harness.py`**

```python
"""AgentHarness — wires checkpointing, interrupt hooks, streaming, and LLM selection."""

import uuid
from collections.abc import AsyncGenerator
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from dev_agent.agent.graph import build_graph
from dev_agent.agent.providers import build_llm
from dev_agent.agent.selector import select_profile
from dev_agent.config import LLMProfile, Settings, get_settings
from dev_agent.tools.registry import build_toolset


class AgentHarness:
    """Manages the compiled agent graph with harness features:
    - MemorySaver checkpointer (thread-scoped conversation memory)
    - Configurable interrupt_before / interrupt_after hooks
    - Multi-LLM profile support with auto-selection
    - Streaming event emission with profile_selected events
    - Human-in-the-loop resume support
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._checkpointer = MemorySaver()
        self._tools = build_toolset(None)

    def new_thread(self) -> str:
        return str(uuid.uuid4())

    def _run_config(self, thread_id: str) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.settings.agent.recursion_limit,
        }

    async def _resolve_profile(self, prompt: str, override: str | None) -> tuple[str, LLMProfile]:
        """Return (profile_name, profile) — runs auto-selection if needed."""
        requested = override or self.settings.agent.profile

        if requested != "auto" and requested in self.settings.profiles:
            return requested, self.settings.profiles[requested]

        # auto mode: use classifier LLM to pick
        selector_profile_name = self.settings.llm_selector.profile
        selector_profile = self.settings.profiles.get(
            selector_profile_name, next(iter(self.settings.profiles.values()))
        )
        classifier_llm = build_llm(selector_profile, tools=[])
        chosen_name = await select_profile(prompt, self.settings.profiles, classifier_llm)
        return chosen_name, self.settings.profiles[chosen_name]

    async def run(
        self,
        prompt: str,
        thread_id: str | None = None,
        workspace: str = ".",
        profile: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream agent events for a given prompt.

        Yields event dicts with keys: type, payload, thread_id.
        Event types: 'profile_selected' | 'token' | 'tool_call' | 'tool_result' | 'done'
        """
        thread_id = thread_id or self.new_thread()

        profile_name, selected_profile = await self._resolve_profile(prompt, profile)
        yield {
            "type": "profile_selected",
            "payload": {"name": profile_name, "model": selected_profile.model,
                        "provider": selected_profile.provider},
            "thread_id": thread_id,
        }

        llm = build_llm(selected_profile, self._tools)
        graph = build_graph(llm, self.settings)
        compiled = graph.compile(
            checkpointer=self._checkpointer,
            interrupt_before=self.settings.harness.interrupt_before or [],
            interrupt_after=self.settings.harness.interrupt_after or [],
        )

        initial_state = {
            "messages": [HumanMessage(content=prompt)],
            "workspace": workspace,
            "tool_calls_count": 0,
            "interrupted": False,
        }

        async for event in compiled.astream_events(
            initial_state, config=self._run_config(thread_id), version="v2"
        ):
            kind = event.get("event", "")
            name = event.get("name", "")
            data = event.get("data", {})

            if kind == "on_chat_model_stream":
                chunk = data.get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield {"type": "token", "payload": chunk.content, "thread_id": thread_id}

            elif kind == "on_tool_start":
                yield {"type": "tool_call",
                       "payload": {"tool": name, "input": data.get("input", {})},
                       "thread_id": thread_id}

            elif kind == "on_tool_end":
                yield {"type": "tool_result",
                       "payload": {"tool": name, "output": str(data.get("output", ""))[:2000]},
                       "thread_id": thread_id}

            elif kind == "on_chain_end" and name == "LangGraph":
                output = data.get("output", {})
                messages = output.get("messages", [])
                last = messages[-1] if messages else None
                final_text = getattr(last, "content", "") if last else ""
                yield {"type": "done", "payload": final_text, "thread_id": thread_id}

    async def resume(
        self,
        thread_id: str,
        value: Any = None,
        profile: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Resume a graph that was interrupted (human-in-the-loop)."""
        profile_name, selected_profile = await self._resolve_profile("", profile)
        llm = build_llm(selected_profile, self._tools)
        graph = build_graph(llm, self.settings)
        compiled = graph.compile(
            checkpointer=self._checkpointer,
            interrupt_before=self.settings.harness.interrupt_before or [],
            interrupt_after=self.settings.harness.interrupt_after or [],
        )

        async for event in compiled.astream_events(
            value, config=self._run_config(thread_id), version="v2"
        ):
            kind = event.get("event", "")
            data = event.get("data", {})
            name = event.get("name", "")

            if kind == "on_chat_model_stream":
                chunk = data.get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield {"type": "token", "payload": chunk.content, "thread_id": thread_id}

            elif kind == "on_tool_start":
                yield {"type": "tool_call",
                       "payload": {"tool": name, "input": data.get("input", {})},
                       "thread_id": thread_id}

            elif kind == "on_tool_end":
                yield {"type": "tool_result",
                       "payload": {"tool": name, "output": str(data.get("output", ""))[:2000]},
                       "thread_id": thread_id}

            elif kind == "on_chain_end" and name == "LangGraph":
                output = data.get("output", {})
                messages = output.get("messages", [])
                last = messages[-1] if messages else None
                yield {"type": "done", "payload": getattr(last, "content", ""), "thread_id": thread_id}

    def get_state(self, thread_id: str):
        return self._checkpointer

    def list_threads(self) -> list[str]:
        try:
            return [c.config["configurable"]["thread_id"] for c in self._checkpointer.list({})]
        except Exception:
            return []
```

- [ ] **Step 2: Verify tests pass**

```bash
pytest tests/ -v
```

Expected: all existing tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/dev_agent/agent/harness.py
git commit -m "feat: wire multi-LLM selector and providers into AgentHarness"
```

---

## Task 9: Update `cli/main.py` — `--profile` flag + `config check`

**Files:**
- Modify: `src/dev_agent/cli/main.py`

- [ ] **Step 1: Add `--profile` option to `run` and `chat`, update `_get_harness`, add `config check`**

Replace `src/dev_agent/cli/main.py` with:

```python
"""Dev Agent CLI — entry point with Typer subcommands."""

import asyncio
import json
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

app = typer.Typer(
    name="dev-agent",
    help="Software development assistant powered by LangGraph Deep Agent.",
    no_args_is_help=True,
)
console = Console()
config_app = typer.Typer(help="Configuration commands.")
app.add_typer(config_app, name="config")


def _get_harness(workspace: str):
    from dev_agent.agent.harness import AgentHarness
    from dev_agent.config import get_settings
    settings = get_settings()
    return AgentHarness(settings), workspace


@app.command("run")
def run_cmd(
    prompt: str = typer.Argument(..., help="Prompt to send to the agent."),
    workspace: str = typer.Option(".", "--workspace", "-w", help="Working directory for tools."),
    thread_id: Optional[str] = typer.Option(None, "--thread", "-t", help="Thread ID (resumes conversation)."),
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="LLM profile name (overrides config)."),
    json_output: bool = typer.Option(False, "--json", help="Emit raw JSON events instead of rendered output."),
):
    """Run the agent with a single prompt and stream the response."""

    async def _run():
        harness, ws = _get_harness(workspace)
        tid = thread_id or harness.new_thread()

        async for event in harness.run(prompt, thread_id=tid, workspace=ws, profile=profile):
            if json_output:
                print(json.dumps(event), flush=True)
                continue

            etype, payload = event["type"], event["payload"]
            if etype == "profile_selected":
                console.print(
                    f"\n[dim][auto → [cyan]{payload['name']}[/] · {payload['model']}][/dim]\n"
                    if (profile is None and harness.settings.agent.profile == "auto") else ""
                )
            elif etype == "token":
                print(payload, end="", flush=True)
            elif etype == "tool_call":
                console.print(f"\n[cyan]⚙ {payload['tool']}[/]", end=" ")
                args_str = ", ".join(f"{k}={v!r}" for k, v in payload.get("input", {}).items())
                console.print(f"[dim]({args_str})[/dim]")
            elif etype == "tool_result":
                out = payload.get("output", "")[:400]
                console.print(f"[dim]  → {out}[/dim]")
            elif etype == "done":
                print()
                console.print(f"\n[dim]thread: {tid}[/dim]")

    asyncio.run(_run())


@app.command("chat")
def chat_cmd(
    workspace: str = typer.Option(".", "--workspace", "-w", help="Working directory for tools."),
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="LLM profile name (overrides config)."),
):
    """Start an interactive chat REPL with the agent."""
    from dev_agent.cli.repl import run_repl

    async def _chat():
        harness, ws = _get_harness(workspace)
        await run_repl(harness, workspace=ws, default_profile=profile)

    asyncio.run(_chat())


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host."),
    port: int = typer.Option(8080, "--port", "-p", help="Bind port."),
    workspace: str = typer.Option(".", "--workspace", "-w", help="Default workspace."),
    secret: str = typer.Option("", "--secret", help="HMAC secret for webhook verification."),
):
    """Start the webhook server for GitHub/GitLab event processing."""
    import uvicorn
    from dev_agent.webhooks.server import create_app
    from dev_agent.config import get_settings

    settings = get_settings()
    if secret:
        settings.webhooks.secret = secret

    harness, ws = _get_harness(workspace)
    web_app = create_app(harness, default_workspace=ws, settings=settings)

    console.print(Panel(
        f"[bold green]Dev Agent Webhook Server[/bold green]\n"
        f"Listening on [cyan]http://{host}:{port}[/cyan]\n"
        f"Workspace: [dim]{ws}[/dim]",
        border_style="green",
    ))
    uvicorn.run(web_app, host=host, port=port, log_level="warning")


@config_app.command("show")
def config_show_cmd():
    """Show the active configuration."""
    import yaml
    from dev_agent.config import get_settings

    settings = get_settings()
    data = {
        "agent": settings.agent.model_dump(),
        "llm_selector": settings.llm_selector.model_dump(),
        "profiles": {name: p.model_dump() for name, p in settings.profiles.items()},
        "harness": settings.harness.model_dump(),
        "webhooks": settings.webhooks.model_dump(),
    }
    console.print(Syntax(yaml.dump(data, default_flow_style=False), "yaml", theme="monokai"))


@config_app.command("check")
def config_check_cmd():
    """Test connectivity for all configured LLM profiles."""
    import asyncio
    from dev_agent.agent.health import check_all
    from dev_agent.config import get_settings

    settings = get_settings()
    console.print("\n[bold]Checking LLM profiles...[/bold]\n")

    statuses = asyncio.run(check_all(settings.profiles))

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("", width=3)
    table.add_column("Profile", style="cyan", no_wrap=True, min_width=12)
    table.add_column("Provider / Model", min_width=30)
    table.add_column("Latency", justify="right", min_width=8)
    table.add_column("Result / Error")

    ok_count = 0
    for s in statuses:
        icon = "[green]✓[/]" if s.ok else "[red]✗[/]"
        provider_model = f"{s.provider} / {s.model}"
        latency = f"{s.latency_ms:.0f}ms" if s.ok else "—"
        result = f'[dim]"{s.snippet}"[/dim]' if s.ok else f"[red]{s.error}[/red]"
        table.add_row(icon, s.name, provider_model, latency, result)
        if s.ok:
            ok_count += 1

    console.print(table)
    total = len(statuses)
    colour = "green" if ok_count == total else "yellow" if ok_count > 0 else "red"
    console.print(f"\n[{colour}]{ok_count}/{total} profiles healthy.[/]\n")


@app.command("tools")
def tools_cmd():
    """List all available agent tools."""
    from dev_agent.tools.registry import list_tools

    table = Table(title="Available Tools", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description")

    for name, desc in list_tools().items():
        table.add_row(name, desc)

    console.print(table)


def main():
    app()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI help renders correctly**

```bash
dev-agent --help
dev-agent config --help
dev-agent run --help
```

Expected: `config` now shows `show` and `check` subcommands; `run` and `chat` show `--profile`.

- [ ] **Step 3: Commit**

```bash
git add src/dev_agent/cli/main.py
git commit -m "feat: add --profile flag and 'config check' subcommand to CLI"
```

---

## Task 10: Update `cli/repl.py` — render profile badge

**Files:**
- Modify: `src/dev_agent/cli/repl.py`

- [ ] **Step 1: Update `_stream_response` in `repl.py` to handle `profile_selected` event**

Find the `_stream_response` function (lines ~43–69). Replace it with:

```python
async def _stream_response(
    harness: "AgentHarness",
    prompt: str,
    thread_id: str,
    workspace: str,
    default_profile: str | None,
) -> str:
    """Stream the agent response, rendering events as they arrive."""
    full_response = ""
    is_auto = default_profile is None and harness.settings.agent.profile == "auto"
    console.print()

    async for event in harness.run(prompt, thread_id=thread_id, workspace=workspace, profile=default_profile):
        etype = event["type"]
        payload = event["payload"]

        if etype == "profile_selected" and is_auto:
            name = payload["name"]
            model = payload["model"]
            console.print(f"[dim][auto → [cyan]{name}[/cyan] · {model}][/dim]")

        elif etype == "token":
            console.print(payload, end="", highlight=False)
            full_response += payload

        elif etype == "tool_call":
            _render_tool_call(payload["tool"], payload.get("input", {}))

        elif etype == "tool_result":
            _render_tool_result(payload["tool"], payload.get("output", ""))

        elif etype == "done":
            if not full_response and payload:
                console.print(Markdown(payload))
                full_response = payload

    console.print()
    return full_response
```

- [ ] **Step 2: Update `run_repl` signature to accept `default_profile`**

Find the `run_repl` function definition (line ~75) and update its signature and the call to `_stream_response`:

```python
async def run_repl(harness: "AgentHarness", workspace: str = ".", default_profile: str | None = None) -> None:
```

And inside the loop, update the call from:
```python
await _stream_response(harness, user_input, thread_id, workspace)
```
to:
```python
await _stream_response(harness, user_input, thread_id, workspace, default_profile)
```

- [ ] **Step 3: Verify tools tests still pass**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/dev_agent/cli/repl.py
git commit -m "feat: render profile badge in REPL when auto mode is active"
```

---

## Task 11: Update `.env.example` and push to GitHub

**Files:**
- Modify: `.env.example`

- [ ] **Step 1: Update `.env.example`**

```bash
# Copy to .env and fill in the keys for the providers you want to use.

# Anthropic (profiles: powerful, balanced)
ANTHROPIC_API_KEY=sk-ant-...

# Groq — free tier available at console.groq.com (profile: fast)
GROQ_API_KEY=gsk_...

# OpenAI (profile: openai)
OPENAI_API_KEY=sk-...

# Google Gemini (profile: google — add to settings.yaml first)
GOOGLE_API_KEY=AIza...

# Ollama runs locally — no key needed. Install: https://ollama.com
# Then: ollama pull qwen2.5-coder:7b

# Optional overrides
DEV_AGENT_WORKSPACE=.
DEV_AGENT_AGENT__PROFILE=auto
DEV_AGENT_LLM_SELECTOR__PROFILE=fast
```

- [ ] **Step 2: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all tests pass (at minimum: test_tools, test_config_profiles, test_providers, test_selector, test_health).

- [ ] **Step 3: Verify CLI commands work end-to-end (no API key required for these)**

```bash
dev-agent --help
dev-agent tools
dev-agent config show
```

Expected: all render without errors, config shows the 5 profiles.

- [ ] **Step 4: Commit and push**

```bash
git add .env.example
git commit -m "docs: update .env.example with all provider keys"
git push
```

---

## Verification

```bash
# Install all provider extras
pip install -e ".[all-llms,dev]"

# Confirm profiles loaded
dev-agent config show

# Health-check all profiles (requires API keys in .env)
dev-agent config check

# Single-shot with explicit profile
dev-agent run "what is 2+2?" --profile fast

# Single-shot with auto mode
dev-agent run "refactor the payments module and add tests" --workspace ~/my-project

# Interactive REPL (auto mode by default)
dev-agent chat

# REPL with forced profile
dev-agent chat --profile powerful

# Full test suite
pytest tests/ -v
```
