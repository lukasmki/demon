from rich.terminal_theme import TerminalTheme
from rich.json import JSON
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.console import Console
from pathlib import Path

import matplotlib.pyplot as plt

from pydantic_ai import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    ThinkingPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)

type History = list[ModelRequest | ModelResponse]


def calc_stats(history: History) -> dict:
    stats = {
        "text_responses": 0,
        "thinking_responses": 0,
        "tool_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    for msg in history:
        if isinstance(msg, ModelResponse):
            stats["text_responses"] += len(
                [p for p in msg.parts if isinstance(p, TextPart)]
            )
            stats["thinking_responses"] += len(
                [p for p in msg.parts if isinstance(p, ThinkingPart)]
            )
            stats["tool_calls"] += len(msg.tool_calls)
            stats["input_tokens"] += msg.usage.input_tokens
            stats["output_tokens"] += msg.usage.output_tokens
    return stats


def plot_stats(chat_stats: dict[str, dict[str, int]]):
    colors = {
        "gemini": "steelblue",
        "chatgpt": "seagreen",
        "claude": "tomato",
        "qwen": "orchid",
    }

    fig, axes = plt.subplots(1, 1, figsize=(12, 7), sharex=True)
    axes.grid(alpha=0.5)

    models = list(chat_stats.keys())
    categories = list(next(iter(chat_stats.values())).keys())
    n_models = len(models)
    n_cats = len(categories)
    x = range(n_cats)
    width = 1 / (n_models + 1)

    for i, model in enumerate(models):
        offsets = [xi + i * width - (n_models - 1) * width / 2 for xi in x]
        values = [chat_stats[model][cat] for cat in categories]
        axes.bar(
            offsets, values, width=width, label=model, color=colors[model], linewidth=1
        )

    axes.set_yscale("log")
    axes.set_xticks(list(x))
    axes.set_xticklabels(categories)
    axes.set_ylabel("Count")
    axes.set_title("Model comparison by stat")
    axes.legend()

    return fig


def make_readable(history: History) -> str:
    console = Console(record=True)

    for event in history:
        for part in event.parts:
            if isinstance(part, ToolCallPart):
                if isinstance(part.args, str):
                    argstr = part.args
                else:
                    argstr = ", ".join([f"{k}={v}" for k, v in part.args.items()])
                call_text = Text()
                call_text.append(part.tool_name, style="bold cyan")
                call_text.append(f"({argstr})", style="dim")
                console.print(
                    Panel(
                        call_text,
                        title="[bold yellow]Tool Call[/bold yellow]",
                        border_style="yellow",
                    )
                )
            elif isinstance(part, ThinkingPart):
                console.print(
                    Panel(
                        Markdown(part.content),
                        title="[bold magenta]Thinking[/bold magenta]",
                        border_style="magenta",
                    )
                )
            elif isinstance(part, TextPart):
                console.print(
                    Panel(
                        Markdown(part.content),
                        title="[bold blue]Output[/bold blue]",
                        border_style="blue",
                    )
                )
            elif isinstance(part, ToolReturnPart):
                content = part.content or ""
                console.print(
                    Panel(
                        Text(content),
                        title="[bold green]Tool Return[/bold green]",
                        border_style="green",
                    )
                )
            else:
                console.print(part)

    return console.export_svg()


if __name__ == "__main__":
    root = Path(__file__).parent
    data = root / "data"
    models = ["gemini", "chatgpt", "claude", "qwen"]

    chat_data: dict[str, History] = {
        model: ModelMessagesTypeAdapter.validate_json(
            (data / model).with_suffix(".json").read_text()
        )
        for model in models
    }

    chat_stats = {model: calc_stats(history) for model, history in chat_data.items()}
    for model, stats in chat_stats.items():
        print(model, stats)

    for model, history in chat_data.items():
        replay = make_readable(history)
        (root / model).with_suffix(".svg").write_text(replay)

    fig = plot_stats(chat_stats)
    fig.savefig(root / "all_stats.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
