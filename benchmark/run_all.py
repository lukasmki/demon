from demon.demon import WalledVdW
from pydantic_ai.providers.openai import OpenAIProvider
from google.genai.types import ThinkingLevel
from anthropic.types.beta import (
    BetaThinkingConfigAdaptiveParam,
)
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.models.openai import (
    OpenAIResponsesModel,
    OpenAIResponsesModelSettings,
    OpenAIChatModel,
    OpenAIChatModelSettings,
)
from dotenv import load_dotenv
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
import numpy as np
from demon import DemonMD
from pathlib import Path
from ase import Atoms, io, units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

np.random.seed(42)


def make_box(N: int = 20, L: float = 40, T: float = 5000, min_dist=4.0) -> Atoms:
    positions = np.zeros((N, 3))
    positions[0] = (L - min_dist) * np.random.random((3)) + (min_dist / 2)
    for i in range(1, N):
        while True:
            ipos = (L - min_dist) * np.random.random((3)) + (min_dist / 2)
            if np.all(np.linalg.norm(positions[:i] - ipos, axis=-1) >= min_dist):
                break
        positions[i] = ipos

    atoms = Atoms(
        numbers=np.full((N,), 18),
        positions=positions,
    )
    atoms.set_cell([L, L, L, 90, 90, 90])
    atoms.set_pbc([False, False, False])
    MaxwellBoltzmannDistribution(atoms, temperature_K=T, force_temp=True)
    return atoms


def run(model: Model, atoms: Atoms, outpath: Path | str, max_steps: int = 500) -> None:
    # Ar-Ar VdW interaction with wall blocking
    atoms.calc = WalledVdW(
        epsilonij=0.0104,
        sigmaij=3.405,
        gamma=0.5 * 3.405,
        attractive=False,
    )

    frames = []

    def add_frame(a: Atoms):
        frames.append(a.copy())

    dyn = DemonMD(
        model,
        atoms,
        timestep=1.0 * units.fs,
        demon_enabled=True,
    )
    dyn.attach(add_frame, 10, atoms)
    dyn.run(steps=max_steps)

    # write
    io.write(Path(outpath).with_suffix(".xyz"), frames)
    Path(outpath).with_suffix(".json").write_bytes(dyn.messages_json)


load_dotenv()
models = {
    # "gemini": GoogleModel(
    #     model_name="gemini-3-flash-preview",
    #     provider="google-gla",
    #     settings=GoogleModelSettings(
    #         google_thinking_config={
    #             "include_thoughts": True,
    #             "thinking_level": ThinkingLevel.MEDIUM,
    #         }
    #     ),
    # ),
    # "chatgpt": OpenAIResponsesModel(
    #     model_name="gpt-5.4",
    #     settings=OpenAIResponsesModelSettings(
    #         openai_reasoning_summary="auto",
    #         openai_reasoning_effort="medium",
    #     ),
    # ),
    # "claude": AnthropicModel(
    #     model_name="claude-sonnet-4-6",
    #     settings=AnthropicModelSettings(
    #         max_tokens=4096 * 8,
    #         anthropic_thinking=BetaThinkingConfigAdaptiveParam(
    #             type="adaptive", display="summarized"
    #         ),
    #         anthropic_effort="medium",
    #     ),
    # ),
    "qwen": OpenAIChatModel(
        model_name="unsloth/Qwen3.5-2B-GGUF",
        provider=OpenAIProvider(base_url="http://localhost:8080"),
        settings=OpenAIChatModelSettings(
            openai_reasoning_effort="medium",
        ),
    ),
}

if __name__ == "__main__":
    root = Path(__file__).parent
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)

    # create super system
    above = make_box(N=30, L=20, T=1000)
    below = make_box(N=30, L=20, T=1000)
    above.translate([0, 0, 20])
    atoms: Atoms = above + below
    atoms.set_cell([20, 20, 40, 90, 90, 90])
    atoms.set_pbc([False, False, False])

    for name, model in models.items():
        print("MODEL", name)
        if (data / name).with_suffix(".json").exists():
            continue
        run(model, atoms.copy(), data / name, 10_000)
