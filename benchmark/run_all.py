from google.genai.types import ThinkingLevel
from anthropic.types.beta import (
    BetaThinkingConfigAdaptiveParam,
    BetaThinkingConfigEnabledParam,
)
from pydantic_ai.profiles.anthropic import AnthropicModelProfile
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from dotenv import load_dotenv
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
import numpy as np
from demon import DemonMD
from pathlib import Path
from ase import Atoms, io
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.calculators.ff import ForceField

np.random.seed(seed=42)


def make_box(N: int = 20, L: float = 40, T: float = 5000) -> Atoms:
    atoms = Atoms(
        numbers=np.full((N,), 2),
        positions=0.98 * L * np.random.random((N, 3)),
    )
    atoms.set_cell([L, L, L, 90, 90, 90])
    atoms.set_pbc([False, False, False])
    MaxwellBoltzmannDistribution(atoms, temperature_K=T, force_temp=True)
    return atoms


def run(model: Model, atoms: Atoms, outpath: Path | str, max_steps: int = 500) -> None:
    atoms.calc = ForceField(morses=[])  # dummy calculator, no particle interactions

    frames = []

    def add_frame(a: Atoms):
        frames.append(a.copy())

    dyn = DemonMD(model, atoms, 1.0, demon_enabled=True)
    dyn.attach(add_frame, 1, atoms)
    dyn.run(steps=max_steps)

    # write
    io.write(Path(outpath).with_suffix(".xyz"), frames)
    Path(outpath).with_suffix(".json").write_bytes(dyn.messages_json)


load_dotenv()
models = {
    "gemini": GoogleModel(
        model_name="gemini-3-flash-preview",
        provider="google-gla",
        settings=GoogleModelSettings(
            google_thinking_config={
                "include_thoughts": True,
                "thinking_level": ThinkingLevel.MEDIUM,
            }
        ),
    ),
    "chatgpt": OpenAIResponsesModel(
        model_name="gpt-5.4",
        settings=OpenAIResponsesModelSettings(
            openai_reasoning_summary="auto",
            openai_reasoning_effort="medium",
        ),
    ),
    "claude": AnthropicModel(
        model_name="claude-sonnet-4-6",
        settings=AnthropicModelSettings(
            max_tokens=4096 * 8,
            anthropic_thinking=BetaThinkingConfigAdaptiveParam(
                type="adaptive", display="summarized"
            ),
            anthropic_effort="medium",
        ),
    ),
}

if __name__ == "__main__":
    root = Path(__file__).parent
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)

    atoms = make_box(N=20, L=40, T=5000)

    for name, model in models.items():
        print("MODEL", name)
        if (data / name).with_suffix(".json").exists():
            continue
        run(model, atoms.copy(), data / name, 500)
