from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIChatModel
from pathlib import Path
from ase.io import read, write
from ase import units, Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from demon.integrator import AgentMD
from ase.calculators.ff import ForceField

root = Path(__file__).parent

model = OpenAIChatModel(
    # model_name="unsloth/Qwen3-0.6B-GGUF",
    model_name="unsloth/Qwen3.5-2B-GGUF",
    provider=OpenAIProvider(base_url="http://127.0.0.1:8080"),
)
atoms = read(root / "water-small.xyz")
assert isinstance(atoms, Atoms)
atoms.calc = ForceField(morses=[])
atoms.set_cell([10, 10, 10, 90, 90, 90])
atoms.center()

MaxwellBoltzmannDistribution(atoms, temperature_K=298)

dyn = AgentMD(model, atoms, timestep=1.0 * units.fs)
dyn.attach(write, 1, root / "simulation.xyz", atoms, format="extxyz", append=True)
dyn.run(steps=10)
