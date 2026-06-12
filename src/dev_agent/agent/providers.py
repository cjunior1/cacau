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
