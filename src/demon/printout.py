from typing import AsyncIterable

from pydantic_ai import (
    PartEndEvent,
    RunContext,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    SystemPromptPart,
)
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from demon.system import System


async def printout(ctx: RunContext[System], events: AsyncIterable):
    _console = Console()

    async for event in events:
        if isinstance(event, PartEndEvent):
            part = event.part
            if isinstance(part, ToolCallPart):
                if isinstance(part.args, str):
                    argstr = part.args
                else:
                    argstr = ", ".join([f"{k}={v}" for k, v in part.args.items()])
                call_text = Text()
                call_text.append(part.tool_name, style="bold cyan")
                call_text.append(f"({argstr})", style="dim")
                _console.print(
                    Panel(
                        call_text,
                        title="[bold yellow]Tool Call[/bold yellow]",
                        border_style="yellow",
                    )
                )
            elif isinstance(part, ThinkingPart):
                _console.print(
                    Panel(
                        Markdown(part.content),
                        title="[bold magenta]Thinking[/bold magenta]",
                        border_style="magenta",
                    )
                )
            elif isinstance(part, TextPart):
                _console.print(
                    Panel(
                        Markdown(part.content),
                        title="[bold blue]Output[/bold blue]",
                        border_style="blue",
                    )
                )
            elif isinstance(part, SystemPromptPart):
                _console.print(
                    Panel(
                        Markdown(part.content),
                        title="[bold green]Output[/bold green]",
                        border_style="green",
                    )
                )
            else:
                print(part)
