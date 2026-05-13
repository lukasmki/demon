from pathlib import Path

import matplotlib.pyplot as plt

from pydantic_ai import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    ThinkingPart,
    TextPart,
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
    colors = {"gemini": "steelblue", "chatgpt": "seagreen", "claude": "tomato"}

    fig, axes = plt.subplots(1, 1, figsize=(12, 7), sharex=True)
    axes.grid(alpha=0.5)

    models = list(chat_stats.keys())
    categories = list(next(iter(chat_stats.values())).keys())
    n_models = len(models)
    n_cats = len(categories)
    x = range(n_cats)
    width = 0.25

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


if __name__ == "__main__":
    root = Path(__file__).parent
    data = root / "data"
    models = ["gemini", "chatgpt", "claude"]

    chat_data: dict[str, History] = {
        model: ModelMessagesTypeAdapter.validate_json(
            (data / model).with_suffix(".json").read_text()
        )
        for model in models
    }

    chat_stats = {model: calc_stats(history) for model, history in chat_data.items()}
    for model, stats in chat_stats.items():
        print(model, stats)

    fig = plot_stats(chat_stats)
    fig.savefig(root / "all_stats.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
