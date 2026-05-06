import numpy as np
from demon import DemonMD
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIChatModel
from pathlib import Path
from ase.io import read, write
from ase import units, Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.calculators.ff import ForceField

root = Path(__file__).parent

model = OpenAIChatModel(
    # model_name="unsloth/Qwen3-0.6B-GGUF",
    model_name="unsloth/Qwen3.5-2B-GGUF",
    provider=OpenAIProvider(base_url="http://localhost:8080"),
)

L = 20  # cell length
N = 20  # number of He atoms

atoms = Atoms(
    numbers=np.full((N,), 2),
    positions=0.98 * 0.5 * L * (2 * np.random.random((N, 3)) - 1),
)
atoms.calc = ForceField(morses=[])
atoms.set_cell([L, L, L, 90, 90, 90])
atoms.set_pbc([False, False, False])
atoms.center()
MaxwellBoltzmannDistribution(atoms, temperature_K=5000, force_temp=True)

dyn = DemonMD(model, atoms, timestep=1)
dyn.attach(write, 1, root / "simulation.xyz", atoms, format="extxyz", append=True)
dyn.run(steps=100)
