from ase.cell import Cell
import numpy as np
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, PartEndEvent
from typing import IO, Annotated, AsyncIterable
from ase.md.md import MolecularDynamics
from ase import Atoms


class System(BaseModel):
    symbols: Annotated[list[str], "Atomic element symbols"]
    positions: Annotated[list[list[float]], "Atomic positions in Angstroms"]
    velocities: Annotated[list[list[float]], "Atomic velocities in Angstrom/fs"]
    cell: Annotated[
        list[float],
        "Unit cell parameters. First three are unit cell vector lengths and second three are angles between them: [len(a), len(b), len(c), angle(b,c), angle(a,c), angle(a,b)]",
    ]

    @classmethod
    def from_atoms(cls, atoms: Atoms) -> "System":
        return cls(
            symbols=atoms.get_chemical_symbols(),
            positions=atoms.get_positions().tolist(),
            velocities=atoms.get_velocities().tolist(),
            cell=atoms.get_cell().cellpar(),
        )

    # def to_atoms(self) -> Atoms:
    #     pass


async def printout(ctx: RunContext[System], events: AsyncIterable):
    async for event in events:
        if isinstance(event, PartEndEvent):
            print(event.part)
            print()


class AgentMD(MolecularDynamics):
    def __init__(
        self,
        model,
        atoms: Atoms,
        timestep: int | float,
        trajectory: str | None = None,
        logfile: IO | str | None = None,
        loginterval: int = 1,
        **kwargs,
    ):
        super().__init__(atoms, timestep, trajectory, logfile, loginterval, **kwargs)
        self.agent = Agent(
            model=model,
            deps_type=System,
            output_type=str,
            # system_prompt=(
            #     "You are a physics simulation integrator. ",
            #     # "You take particle positions, velocities, and unit cell dimensions as input.",
            #     "You update the positions according to your understanding of physics.",
            # ),
        )
        self.register_tools(self.agent)
        self.messages = []

    def register_tools(self, agent: Agent[System, str]) -> None:
        @agent.tool
        def get_system(ctx: RunContext[System]) -> str:
            "Returns the current state of the system"
            return ctx.deps.model_dump_json()

        @agent.tool
        def set_positions(
            ctx: RunContext[System],
            positions: Annotated[list[list[float]], "Atomic positions"],
        ) -> None:
            "Sets the positions of the atoms"
            assert len(positions) == len(
                ctx.deps.positions
            ), f"Positions has wrong size, natoms={len(ctx.deps.positions)}"
            setattr(ctx.deps, "positions", positions)

        @agent.tool
        def set_cell(
            ctx: RunContext[System],
            values: Annotated[
                list[float],
                "Unit cell parameters. First three are unit cell vector lengths and second three are angles between them: [len(a), len(b), len(c), angle(b,c), angle(a,c), angle(a,b)]",
            ],
        ) -> None:
            "Sets the unit cell vectors"
            setattr(ctx.deps, "cell", values)

    def step(self, forces=None):
        atoms = self.atoms
        natoms = len(atoms)

        s: list[str] = atoms.get_chemical_symbols()
        v: np.ndarray = atoms.get_velocities()
        x: np.ndarray = atoms.get_positions()
        c: Cell = atoms.get_cell()

        step_deps = System(
            symbols=s,
            positions=x.tolist(),
            velocities=v.tolist(),
            cell=c.cellpar(),
        )

        result = self.agent.run_sync(
            user_prompt=(
                f"Integrate the simulation forward in time by {self.dt} fs by updating the atomic positions. ",
                # "You are provided a few tools for getting and setting properties of the system. ",
                "When you are finished and ready for the next step, output 'Step complete'.",
            ),
            deps=step_deps,
            output_type=str,
            event_stream_handler=printout,
            # message_history=self.messages,
        )
        print(result.output)
        # self.messages = result.all_messages()

        # full step in position
        new_x = np.array(step_deps.positions)

        atoms.set_positions(new_x)
        atoms.set_momenta(self.masses * (atoms.get_positions() - x) / self.dt)
