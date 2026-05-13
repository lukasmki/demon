import numpy as np
from ase.cell import Cell
from pydantic import BaseModel, PrivateAttr
from pydantic_ai import (
    Agent,
    RunContext,
    PartEndEvent,
    ToolCallPart,
    ThinkingPart,
    TextPart,
    UsageLimits,
    UsageLimitExceeded,
)
from typing import IO, Annotated, AsyncIterable, Literal
from ase.md.md import MolecularDynamics
from ase import Atoms, units
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

_console = Console()


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
            cell=atoms.get_cell().cellpar().tolist(),
            temp_above=float(T_a),
            temp_below=float(T_b),
            door=atoms.info.get("door", "closed"),
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
        self.cell = cell.cellpar().tolist()
        self.temp_above = float(T_a)
        self.temp_below = float(T_b)
        self.door = atoms.info.get("door", "closed")
        self._atoms = atoms


async def printout(ctx: RunContext[System], events: AsyncIterable):
    async for event in events:
        if isinstance(event, PartEndEvent):
            part = event.part
            if isinstance(part, ToolCallPart):
                if isinstance(part.args, str):
                    argstr = part.args
                else:
                    argstr = ", ".join([f"{k}={v}" for k, v in part.args.items()])
                call_text = Text()
                call_text.append(part.tool_name, style="bold cyan")
                call_text.append(f"({argstr})", style="dim")
                _console.print(
                    Panel(
                        call_text,
                        title="[bold yellow]Tool Call[/bold yellow]",
                        border_style="yellow",
                    )
                )
            elif isinstance(part, ThinkingPart):
                _console.print(
                    Panel(
                        Markdown(part.content),
                        title="[bold magenta]Thinking[/bold magenta]",
                        border_style="magenta",
                    )
                )
            elif isinstance(part, TextPart):
                _console.print(
                    Panel(
                        Markdown(part.content),
                        title="[bold blue]Output[/bold blue]",
                        border_style="blue",
                    )
                )
            else:
                print(part)


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
                "## Setup\n"
                "You are controlling a molecular dynamics simulation. "
                "N particles move freely inside a cubic box of side length L, bouncing elastically off the walls. "
                "The box is split into two halves by an invisible wall (the 'door') at z = L/2: "
                "the ABOVE half (z > L/2) and the BELOW half (z < L/2).\n\n"
                "## Your goal\n"
                "Maximize the temperature difference |T_above - T_below|. "
                "Temperature is proportional to the average kinetic energy of particles in each half. "
                "You win by sorting fast (hot) particles to one side and slow (cold) particles to the other.\n\n"
                "## The door rules\n"
                "- When the door is OPEN: particles pass freely between halves.\n"
                "- When the door is CLOSED: particles cannot cross z = L/2 and bounce back elastically.\n"
                "- A particle's 'home' half is determined by which side it was on when the door last acted on it — "
                "closing the door traps each particle in whichever half it currently occupies.\n\n"
                "## Available tools\n"
                "- `get_system`: returns the full state — positions, velocities, and current temperatures T_above and T_below.\n"
                "- `get_door_state`: returns whether the door is currently open or closed.\n"
                "- `set_door_state`: open or close the door.\n"
                "- `wait(steps)`: advance the simulation by the given number of time steps without changing the door. "
                "Use this to let particles travel toward (or away from) the door before acting.\n\n"
                "## Strategy hints\n"
                "The optimal agent watches individual particle velocities and positions, then:\n"
                "1. Opens the door briefly to let a fast particle cross from BELOW to ABOVE (or a slow one from ABOVE to BELOW).\n"
                "2. Closes the door immediately after to trap the temperature asymmetry.\n"
                "A simpler but effective heuristic: if T_below > T_above, open the door so heat flows upward on average; "
                "once T_above > T_below, close the door to lock in the difference. "
                "Repeat, always reinforcing whichever half is already hotter.\n\n"
                "## Termination\n"
                "When you are satisfied with the achieved temperature difference, call the `finished` tool to release control of the simulation. "
                "You do NOT need to reach a perfect outcome — stop when further improvement seems unlikely."
            ),
        )
        self.register_tools(self.agent)
        self.messages = []
        self.messages_json: bytes = bytes()
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
            ctx.deps._atoms.info["door"] = state
            return f"The door is {state}."

        @agent.tool
        def wait(ctx: RunContext[System], steps: int) -> str:
            "Advance the simulation by number of time steps"
            for _ in range(steps):
                a = ctx.deps._atoms
                a.arrays["door_system"] = np.where(
                    a.get_positions()[:, 2] >= 0.5 * a.get_cell()[2, 2], 1, -1
                )
                self.verlet_step(a)
                self.physics_step(a, door=ctx.deps.door)
                ctx.deps.update_from_atoms(a)
                a.info.update(
                    {
                        "door": ctx.deps.door,
                        "T_a": ctx.deps.temp_above,
                        "T_b": ctx.deps.temp_below,
                    }
                )
                self.nsteps += 1
                self.call_observers()
                # if self.nsteps >= self.max_steps:
                #     finished()
            return f"Simulation advanced by {steps}."

        @agent.tool_plain
        def finished() -> None:
            "Declare you are finished with the task."
            self.demon_enabled = False

    def verlet_step(self, atoms: Atoms, forces=None):
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

        # elastic collision with the door wall (xy-plane, z=L/2) when closed
        if door == "closed":
            system = atoms.arrays["door_system"]
            # above-system atoms that crossed to z<L/2
            mask_above = (system == 1) & (pos[:, 2] < 0.5 * cell_matrix[2, 2])
            vel[mask_above, 2] *= -1
            pos[mask_above, 2] = cell_matrix[2, 2] - pos[mask_above, 2]
            # below-system atoms that crossed to z>L/2
            mask_below = (system == -1) & (pos[:, 2] > 0.5 * cell_matrix[2, 2])
            vel[mask_below, 2] *= -1
            pos[mask_below, 2] = cell_matrix[2, 2] - pos[mask_below, 2]
        atoms.set_positions(pos)
        atoms.set_velocities(vel)
        return forces

    def step(self, forces=None):
        atoms = self.atoms
        step_deps = System.from_atoms(atoms)
        atoms.arrays["door_system"] = np.where(
            atoms.get_positions()[:, 2] >= 0.5 * atoms.get_cell()[2, 2], 1, -1
        )
        forces = self.verlet_step(atoms, forces=forces)
        forces = self.physics_step(atoms, step_deps.door, forces=forces)
        step_deps.update_from_atoms(atoms)
        atoms.info.update(
            {
                "door": step_deps.door,
                "T_a": step_deps.temp_above,
                "T_b": step_deps.temp_below,
            }
        )
        if self.demon_enabled:
            try:
                result = self.agent.run_sync(
                    user_prompt=(f"STEP: {self.nsteps}"),
                    deps=step_deps,
                    output_type=str,
                    event_stream_handler=printout,
                    message_history=self.messages,
                    usage_limits=UsageLimits(request_limit=100),
                )
            except UsageLimitExceeded:
                self.demon_enabled = False
            self.messages = result.all_messages()
            self.messages_json = result.all_messages_json()

        return forces
