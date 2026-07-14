from argparse import ArgumentParser
from pathlib import Path

from ase import io
from pydantic_ai.models.openai import (
    OpenAIChatModel,
    OpenAIChatModelSettings,
)
from pydantic_ai.providers.openai import OpenAIProvider

from demon.demon import DemonMD
from demon.system import SystemJSON, SystemXYZ, SystemDebug
from demon.utils import setup_system


def main(
    name: str,
    model_name: str,
    system: str,
    outpath: Path,
    N: int,
    V: float,
    T: float,
    max_steps: int,
    diff: int,
    seed: int,
    host: str,
    port: int,
):
    if outpath.with_suffix(".json").exists():
        print(f"Data for {name} already exists: {outpath}")
        return None

    difficulty = {
        0: (False, False, False),
        1: (True, False, False),
        2: (True, False, True),
        3: (True, True, True),
    }
    eq_constraint, attract, repulse = difficulty.get(diff, difficulty[1])

    systems = {
        "json": SystemJSON,
        "xyz": SystemXYZ,
        "debug": SystemDebug,
    }

    atoms = setup_system(N, V, T, VA=attract, VR=repulse, seed=seed)
    model = OpenAIChatModel(
        model_name=model_name,
        provider=OpenAIProvider(base_url=f"http://{host}:{port}/v1"),
        settings=OpenAIChatModelSettings(
            openai_reasoning_effort="medium",
        ),
    )

    frames = []
    dyn = DemonMD(model, atoms, system=systems[system])
    dyn.attach(lambda x: frames.append(x.copy()), 10, atoms)
    dyn.run(steps=max_steps)

    io.write(Path(outpath).with_suffix(".xyz"), frames)
    Path(str(outpath) + ".json").write_bytes(dyn.messages_json)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--name", type=str, default="", required=False)
    parser.add_argument("--system", type=str, default="json", required=False)
    parser.add_argument("--diff", type=int, default=1, required=False)
    parser.add_argument("--natoms", type=int, default=30, required=False)
    parser.add_argument("--volume", type=float, default=8000, required=False)
    parser.add_argument("--temp", type=float, default=5000, required=False)
    parser.add_argument("--max-steps", type=int, default=10_000, required=False)
    parser.add_argument("--seed", type=int, default=1337, required=False)
    parser.add_argument("--host", type=str, default="localhost", required=False)
    parser.add_argument("--port", type=int, default=8000, required=False)
    args = parser.parse_args()

    print("Running with arguments:", args)

    root = Path(__file__).parent
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    name = args.name or args.model
    outpath = Path(data / name)

    main(
        name=name,
        model_name=args.model,
        system=args.system,
        outpath=outpath,
        N=args.natoms,
        V=args.volume,
        T=args.temp,
        max_steps=args.max_steps,
        diff=args.diff,
        seed=args.seed,
        host=args.host,
        port=args.port,
    )
