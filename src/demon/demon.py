from typing import IO, Literal

import numpy as np
from ase import Atoms, units
from ase.cell import Cell
from ase.md.md import MolecularDynamics
from pydantic_ai import Agent, RunContext, UsageLimitExceeded, UsageLimits
from pydantic_ai.models import Model

from demon.printout import printout
from demon.system import System


class DemonMD(MolecularDynamics):
    def __init__(
        self,
        model: Model,
        atoms: Atoms,
        system: type[System],
        timestep: int | float = units.fs,
        demon_enabled: bool = True,
        trajectory: str | None = None,
        logfile: IO | str | None = None,
        loginterval: int = 1,
        **kwargs,
    ):
        super().__init__(atoms, timestep, trajectory, logfile, loginterval, **kwargs)
        self.agent = Agent(
            model=model,
            deps_type=system,
            system_prompt=system.system_prompt(),
            output_type=str,
        )
        self.register_tools(self.agent)

        self.demon_enabled = demon_enabled
        self.system_cls = system
        self.messages: list = []
        self.messages_json: bytes = bytes()

    def register_tools(self, agent: Agent[System, str]) -> None:
        @agent.tool
        def get_system(ctx: RunContext[System]) -> str:
            "Gets the current state of the system"
            return ctx.deps.serialize()

        @agent.tool
        def get_door_state(ctx: RunContext[System]) -> str:
            "Gets the current state of the door"
            return f"The door is {ctx.deps.door}."

        @agent.tool
        def set_door_state(
            ctx: RunContext[System], state: Literal["open", "closed"]
        ) -> str:
            "Sets the door to the open or closed state"
            ctx.deps.set_door_state(state)
            return f"The door is {state}."

        @agent.tool
        def wait(ctx: RunContext[System], steps: int) -> str:
            "Advance the simulation by a number of time steps"
            atoms = ctx.deps.get_atoms()
            for _ in range(steps):
                self.physics_step(atoms)
                ctx.deps.update(atoms)
                self.nsteps += 1
                self.call_observers()
            return f"Simulation advanced by {steps} step(s)."

        @agent.tool_plain
        def finished() -> None:
            "Declare you are finished with the task. After calling this tool, produce your final response to release control of the simulation."
            self.demon_enabled = False

    def physics_step(self, atoms: Atoms, forces=None):
        ## Step 0: Save subsystem before Verlet step
        atoms.arrays["door_system"] = np.where(
            atoms.get_positions()[:, 2] >= 0.5 * atoms.get_cell()[2, 2], 1, -1
        )

        ## Step 1
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

        ## Step 2
        # Elastic collision with walls
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
        if atoms.info["door"] == "closed":
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
        system = self.system_cls.from_atoms(atoms)
        if self.demon_enabled:
            try:
                result = self.agent.run_sync(
                    user_prompt=self.system_cls.user_prompt(self),
                    deps=system,
                    output_type=str,
                    event_stream_handler=printout,
                    message_history=self.messages,
                    usage_limits=UsageLimits(request_limit=100),
                )
                self.messages = result.all_messages()
                self.messages_json = result.all_messages_json()
            except UsageLimitExceeded:
                self.demon_enabled = False
        self.nsteps -= 1
        return forces
