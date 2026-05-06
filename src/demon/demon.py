from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, PartEndEvent
from typing import IO, Annotated, AsyncIterable, Literal
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

    # the demon's door
    door: Annotated[
        Literal["open", "closed"], "state of the door between the two systems"
    ] = "open"

    @classmethod
    def from_atoms(cls, atoms: Atoms) -> "System":
        return cls(
            symbols=atoms.get_chemical_symbols(),
            positions=atoms.get_positions().tolist(),
            velocities=atoms.get_velocities().tolist(),
            cell=atoms.get_cell().cellpar(),
        )


async def printout(ctx: RunContext[System], events: AsyncIterable):
    async for event in events:
        if isinstance(event, PartEndEvent):
            print(event.part)
            print()


class DemonMD(MolecularDynamics):
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
        )
        self.register_tools(self.agent)
        self.messages = []

    def register_tools(self, agent: Agent[System, str]) -> None:
        @agent.tool
        def get_system(ctx: RunContext[System]) -> str:
            "Returns the current state of the system"
            return ctx.deps.model_dump_json()

        @agent.tool
        def get_door_state(ctx: RunContext[System]) -> str:
            "Gets the current state of the door"
            return f"The door is {ctx.deps.door}."

        @agent.tool
        def set_door_state(
            ctx: RunContext[System], state: Literal["open", "closed"]
        ) -> str:
            "Sets the door to the open or closed state"
            ctx.deps.door = state
            return f"The door is {state}."

    def verlet_step(self, atoms, forces=None):
        # VelocityVerlet.step()
        if forces is None:
            forces = atoms.get_forces(md=True)
        p = atoms.get_momenta()
        p += 0.5 * self.dt * forces
        masses = atoms.get_masses()[:, None]
        r = atoms.get_positions()
        atoms.set_positions(r + self.dt * p / masses)
        if atoms.constraints:
            p = (atoms.get_positions() - r) * masses / self.dt
        atoms.set_momenta(p, apply_constraint=False)
        forces = atoms.get_forces(md=True)
        atoms.set_momenta(atoms.get_momenta() + 0.5 * self.dt * forces)
        return forces

    def step(self, forces=None):
        atoms = self.atoms
        step_deps = System.from_atoms(atoms)

        result = self.agent.run_sync(
            user_prompt=(
                "You control a door that ",
                "You must decide whether to leave the door in its current state or change the state of the door.",
            ),
            deps=step_deps,
            output_type=str,
            event_stream_handler=printout,
            # message_history=self.messages,
        )
        print(result.output)
        # self.messages = result.all_messages()

        forces = self.verlet_step(atoms, forces=forces)
        return forces
