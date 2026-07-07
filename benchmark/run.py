from argparse import ArgumentParser
from demon.demon import WalledVdW
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import (
    OpenAIChatModel,
    OpenAIChatModelSettings,
)
from pydantic_ai.models import Model
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--max-steps", type=int, default=10_000, required=False)
    parser.add_argument("--url", default="http://localhost", required=False)
    parser.add_argument("--port", type=int, default=8080, required=False)
    args = parser.parse_args()

    root = Path(__file__).parent
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)

    # create super system
    above = make_box(N=30, L=20, T=5000)
    below = make_box(N=30, L=20, T=5000)
    above.translate([0, 0, 20])
    atoms: Atoms = above + below
    atoms.set_cell([20, 20, 40, 90, 90, 90])
    atoms.set_pbc([False, False, False])

    print("Running", args.name)
    if (data / args.name).with_suffix(".json").exists():
        print(f"Data for {args.name} already exists")

    model = OpenAIChatModel(
        model_name=args.model,
        provider=OpenAIProvider(base_url=f"{args.url}:{args.port}"),
        settings=OpenAIChatModelSettings(
            openai_reasoning_effort="medium",
        ),
    )

    run(model, atoms.copy(), data / args.name, 10_000)
