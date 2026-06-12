"""Profile health checker — tests each LLM profile with a simple call."""

import asyncio
import time
from dataclasses import dataclass

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
