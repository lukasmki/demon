from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
import numpy as np
from demon import DemonMD
from pathlib import Path
from ase.io import write
from ase import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.calculators.ff import ForceField

root = Path(__file__).parent

# model = OpenAIChatModel(
#     # model_name="unsloth/Qwen3-0.6B-GGUF",
#     model_name="unsloth/Qwen3.5-2B-GGUF",
#     provider=OpenAIProvider(base_url="http://localhost:8080"),
# )

model = GoogleModel(
    "gemini-3-flash-preview",
    provider="google-gla",
    settings=GoogleModelSettings(
        google_thinking_config={
            "include_thoughts": True,
        }
    ),
)

L = 40  # cell length
N = 20  # number of He atoms

atoms = Atoms(
    numbers=np.full((N,), 2),
    positions=0.98 * L * np.random.random((N, 3)),
)
atoms.calc = ForceField(morses=[])
atoms.set_cell([L, L, L, 90, 90, 90])
atoms.set_pbc([False, False, False])
# atoms.center(about=(0.0, 0.0, 0.0))
MaxwellBoltzmannDistribution(atoms, temperature_K=5000, force_temp=True)

dyn = DemonMD(model, atoms, timestep=1)
dyn.demon_enabled = True
dyn.attach(write, 1, root / "simulation.xyz", atoms, format="extxyz", append=True)
dyn.run(steps=500)
(root / "trajectory.json").write_bytes(dyn.messages_json)
