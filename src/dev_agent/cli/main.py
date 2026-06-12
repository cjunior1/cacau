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
