import numpy as np
from ase.cell import Cell
from pydantic import BaseModel, PrivateAttr
from pydantic_ai import Agent, RunContext, PartEndEvent
from typing import IO, Annotated, AsyncIterable, Literal
from ase.md.md import MolecularDynamics
from ase import Atoms, units


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
    temp_above: Annotated[float, "Temperature (K) of particles above z=L/2"]
    temp_below: Annotated[float, "Temperature (K) of particles below z=L/2"]

    # hidden
    _atoms: Atoms = PrivateAttr()

    @classmethod
    def from_atoms(cls, atoms: Atoms) -> "System":
        cell = atoms.get_cell()
        pos = atoms.get_positions()
        mom = atoms.get_momenta()
        vel = atoms.get_velocities()
        above = pos[:, 2] > 0.5 * cell[2, 2]
        below = pos[:, 2] < 0.5 * cell[2, 2]
        T_a = np.vdot(mom[above], vel[above]) / (3 * np.sum(above) * units.kB + 1e-8)
        T_b = np.vdot(mom[below], vel[below]) / (3 * np.sum(below) * units.kB + 1e-8)

        inst = cls(
            symbols=atoms.get_chemical_symbols(),
            positions=atoms.get_positions().tolist(),
            velocities=atoms.get_velocities().tolist(),
            cell=atoms.get_cell().cellpar(),
            temp_above=float(T_a),
            temp_below=float(T_b),
            door=atoms.info.get("door", "open"),
        )
        inst._atoms = atoms
        return inst

    def update_from_atoms(self, atoms: Atoms) -> None:
        cell = atoms.get_cell()
        pos = atoms.get_positions()
        mom = atoms.get_momenta()
        vel = atoms.get_velocities()
        above = pos[:, 2] > 0.5 * cell[2, 2]
        below = pos[:, 2] < 0.5 * cell[2, 2]
        T_a = np.vdot(mom[above], vel[above]) / (3 * np.sum(above) * units.kB + 1e-8)
        T_b = np.vdot(mom[below], vel[below]) / (3 * np.sum(below) * units.kB + 1e-8)

        self.symbols = atoms.get_chemical_symbols()
        self.positions = pos.tolist()
        self.velocities = vel.tolist()
        self.cell = cell.cellpar()
        self.temp_above = float(T_a)
        self.temp_below = float(T_b)
        self.door = atoms.info.get("door", "open")
        self._atoms = atoms


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
        demon_enabled: bool = False,
        trajectory: str | None = None,
        logfile: IO | str | None = None,
        loginterval: int = 1,
        **kwargs,
    ):
        super().__init__(atoms, timestep, trajectory, logfile, loginterval, **kwargs)
        self.agent = Agent(
            model=model,
            deps_type=System,
            instructions=(
                "You are given a box of particles that collide elastically against the walls of a box with side lengths, L. ",
                "You control a door that divides the top and bottom halves of the box along the z=L/2 plane. ",
                "Your goal is to maximize the temperature difference between the halves by controlling the flow of particles between the two subsystems. ",
                "You must decide whether to leave the door in its current state or change the state of the door. ",
                # "Your final result should be the number of time steps (>= 1) to advance the simulation before you make your next decision to open or close the door. ",
                # "When you are finished making the decision, call the `final_result` tool with the number of steps to wait.",
                "If you are satisfied with the acheieved temperature difference, return your use the `final_result` tool with a value of 42.",
            ),
        )
        self.register_tools(self.agent)
        self.messages = []
        self.messages_json: bytes = bytes()
        self.timer = 0
        self.demon_enabled: bool = demon_enabled

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

        @agent.tool
        def wait(ctx: RunContext[System], steps: int) -> str:
            "Advance the simulation by number of time steps"
            for _ in range(steps):
                self.verlet_step(ctx.deps._atoms)
                self.physics_step(ctx.deps._atoms, door=ctx.deps.door)
                self.nsteps += 1
                self.call_observers()
                # print("STEP", ctx.deps._atoms.positions)
            ctx.deps.update_from_atoms(ctx.deps._atoms)
            return f"Simulation advanced by {steps}."

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

    def physics_step(self, atoms: Atoms, door: Literal["open", "closed"], forces=None):
        # walls
        cell: Cell = atoms.get_cell()
        pos: np.ndarray = atoms.get_positions()
        vel: np.ndarray = atoms.get_velocities()
        atoms.arrays["door_system"] = np.where(pos[:, 2] >= cell.cellpar()[2], 1, -1)

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
        if door == "closed":
            system = atoms.arrays["door_system"]
            # above-system atoms that crossed to z<L/2
            mask_above = (system == 1) & (pos[:, 2] < 0.5 * cell_matrix[2, 2])
            vel[mask_above, 2] *= -1
            pos[mask_above, 2] = cell_matrix[2, 2] - pos[mask_above, 2]
            # below-system atoms that crossed to z>L/2
            mask_below = (system == -1) & (pos[:, 2] > 0.5 * cell_matrix[2, 2])
            vel[mask_below, 2] *= -1
            pos[mask_below, 2] = 0.5 * cell_matrix[2, 2] - pos[mask_below, 2]
        atoms.set_positions(pos)
        atoms.set_velocities(vel)
        return forces

    def step(self, forces=None):
        print("STEP:", self.nsteps)
        atoms = self.atoms
        step_deps = System.from_atoms(atoms)

        if self.demon_enabled:
            self.timer -= 1
            while self.timer <= 0:
                result = self.agent.run_sync(
                    user_prompt=(f"STEP: {self.nsteps}"),
                    deps=step_deps,
                    output_type=int,
                    event_stream_handler=printout,
                    message_history=self.messages,
                )
                print("OUTPUT:", result.output)
                if result.output == 42:
                    print("AGENT DECLARED COMPLETE")
                    self.demon_enabled = False

                self.messages = result.all_messages()
                self.messages_json = result.all_messages_json()
                self.timer = result.output

        atoms.info.update(
            {
                "door": step_deps.door,
                "T_a": step_deps.temp_above,
                "T_b": step_deps.temp_below,
            }
        )
        forces = self.verlet_step(atoms, forces=forces)
        forces = self.physics_step(atoms, step_deps.door, forces=forces)
        return forces
