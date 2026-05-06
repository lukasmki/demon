import numpy as np
from ase.cell import Cell
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
        Literal["open", "closed"], "State of the door between the two systems"
    ]

    @classmethod
    def from_atoms(cls, atoms: Atoms) -> "System":
        pos = atoms.get_positions()
        mom = atoms.get_momenta()
        vel = atoms.get_velocities()
        # ekin = 0.5 * np.vdot(mom, vel)
        # 2 * ekin / (3 * N * units.kB)

        return cls(
            symbols=atoms.get_chemical_symbols(),
            positions=atoms.get_positions().tolist(),
            velocities=atoms.get_velocities().tolist(),
            cell=atoms.get_cell().cellpar(),
            door=atoms.info.get("door", "closed"),
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
        self.timer = 0

    def register_tools(self, agent: Agent[System, str]) -> None:
        @agent.tool
        def get_system(ctx: RunContext[System]) -> str:
            "Gets the current state of the system"
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
        print("STEP:", self.nsteps)
        atoms = self.atoms
        step_deps = System.from_atoms(atoms)

        self.timer -= 1
        while self.timer <= 0:
            result = self.agent.run_sync(
                user_prompt=(
                    f"STEP: {self.nsteps}\n\n",
                    "You control a door that divides the system along the xy-plane. ",
                    "Your goal is to create a hot and a cold subsystem above and below the xy-plane by controlling which particles are allowed to cross when the door is open and closed. ",
                    "You must decide whether to leave the door in its current state or change the state of the door. ",
                    "Your final result should be the number of time steps (>= 1) to advance the simulation before you make your next decision to open or close. ",
                    "When you are finished, call the `final_result` tool with the number of steps to wait.",
                ),
                deps=step_deps,
                output_type=int,
                event_stream_handler=printout,
                message_history=self.messages,
            )
            print(result.output)
            self.messages = result.all_messages()
            self.timer = result.output

        atoms.info.update({"door": step_deps.door})
        atoms.arrays["door_system"] = np.where(atoms.get_positions()[:, 2] >= 0, 1, -1)
        forces = self.verlet_step(atoms, forces=forces)

        # walls
        cell: Cell = atoms.get_cell()
        pos: np.ndarray = atoms.get_positions()
        vel: np.ndarray = atoms.get_velocities()

        # elastic collision with cell boundaries via fractional coordinates
        cell_matrix = cell.array  # (3, 3), rows are cell vectors
        inv_cell = np.linalg.inv(cell_matrix)
        frac = pos @ inv_cell  # (N, 3) fractional coords

        for i in range(3):
            cell_hat = cell_matrix[i] / np.linalg.norm(cell_matrix[i])
            v_proj = vel @ cell_hat  # (N,) velocity component along cell vector i

            mask_lo = frac[:, i] < 0
            vel[mask_lo] -= 2 * v_proj[mask_lo, None] * cell_hat
            frac[mask_lo, i] *= -1

            mask_hi = frac[:, i] > 1
            vel[mask_hi] -= 2 * v_proj[mask_hi, None] * cell_hat
            frac[mask_hi, i] = 2 - frac[mask_hi, i]

        pos = frac @ cell_matrix

        # elastic collision with the door wall (xy-plane, z=0) when closed
        if step_deps.door == "closed":
            system = atoms.arrays["door_system"]
            # above-system atoms that crossed to z<0
            mask_above = (system == 1) & (pos[:, 2] < 0)
            vel[mask_above, 2] *= -1
            pos[mask_above, 2] *= -1
            # below-system atoms that crossed to z>0
            mask_below = (system == -1) & (pos[:, 2] > 0)
            vel[mask_below, 2] *= -1
            pos[mask_below, 2] *= -1

        atoms.set_positions(pos)
        atoms.set_velocities(vel)

        return forces
