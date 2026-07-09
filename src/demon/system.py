from abc import ABC, abstractmethod
from typing import Annotated, Literal

import numpy as np
from ase import Atoms, units
from ase.md.md import MolecularDynamics
from pydantic import BaseModel, PrivateAttr


class System(ABC, BaseModel):
    symbols: Annotated[list[str], "Atom symbols"]
    position: Annotated[list[list[float]], "Cartesian positions"]
    velocity: Annotated[list[list[float]], "Cartesian velocities"]
    cell: Annotated[list[list[float]], "Unit cell vectors"]
    door: Annotated[
        Literal["open", "closed"], "State of the door between the two systems"
    ]
    T_above: Annotated[float, "Temperature (K) of particles above z=L/2"]
    T_below: Annotated[float, "Temperature (K) of particles below z=L/2"]
    N_above: Annotated[int, "Number of particles above z=L/2"]
    N_below: Annotated[int, "Number of particles below z=L/2"]

    _atoms: Atoms = PrivateAttr()

    def update(self, atoms: Atoms) -> None:
        cell: np.ndarray = atoms.get_cell().array
        pos: np.ndarray = atoms.get_positions()
        mom = atoms.get_momenta()
        vel: np.ndarray = atoms.get_velocities()
        above = pos[:, 2] > 0.5 * cell[2, 2]
        below = pos[:, 2] < 0.5 * cell[2, 2]
        T_a = np.vdot(mom[above], vel[above]) / (3 * np.sum(above) * units.kB + 1e-8)
        T_b = np.vdot(mom[below], vel[below]) / (3 * np.sum(below) * units.kB + 1e-8)
        N_a = np.sum(above)
        N_b = np.sum(below)
        self.symbols = atoms.get_chemical_symbols()
        self.position = pos.tolist()
        self.velocity = vel.tolist()
        self.cell = cell.tolist()
        self.T_above = float(T_a)
        self.T_below = float(T_b)
        self.N_above = int(N_a)
        self.N_below = int(N_b)
        self.door = atoms.info.get("door", "closed")
        atoms.info.update(
            {
                "door": self.door,
                "T_a": float(T_a),
                "T_b": float(T_b),
                "N_a": int(N_a),
                "N_b": int(N_b),
            }
        )
        self._atoms = atoms

    @classmethod
    def from_atoms(cls, atoms: Atoms) -> "System":
        cell: np.ndarray = atoms.get_cell().array
        pos: np.ndarray = atoms.get_positions()
        mom = atoms.get_momenta()
        vel: np.ndarray = atoms.get_velocities()
        above = pos[:, 2] > 0.5 * cell[2, 2]
        below = pos[:, 2] < 0.5 * cell[2, 2]
        T_a = np.vdot(mom[above], vel[above]) / (3 * np.sum(above) * units.kB + 1e-8)
        T_b = np.vdot(mom[below], vel[below]) / (3 * np.sum(below) * units.kB + 1e-8)
        N_a = np.sum(above)
        N_b = np.sum(below)
        inst = cls(
            symbols=atoms.get_chemical_symbols(),
            position=pos.tolist(),
            velocity=vel.tolist(),
            cell=cell.tolist(),
            T_above=float(T_a),
            T_below=float(T_b),
            N_above=int(N_a),
            N_below=int(N_b),
            door=atoms.info.get("door", "closed"),
        )
        atoms.info.update(
            {
                "door": inst.door,
                "T_a": float(T_a),
                "T_b": float(T_b),
                "N_a": int(N_a),
                "N_b": int(N_b),
            }
        )
        inst._atoms = atoms
        return inst

    def set_door_state(self, state: Literal["open", "closed"]):
        self.door = state
        self._atoms.info["door"] = state

    def get_atoms(self, copy: bool = False) -> Atoms:
        if copy:
            return self._atoms.copy()
        return self._atoms

    @staticmethod
    @abstractmethod
    def system_prompt() -> str: ...

    @staticmethod
    @abstractmethod
    def user_prompt(dyn: MolecularDynamics) -> str: ...

    @abstractmethod
    def serialize(self) -> str: ...


class SystemJSON(System):
    @staticmethod
    def system_prompt() -> str:
        return """
## Setup
You are running a molecular dynamics simulation. The simulation box spans z from 0 to 2L and is divided into two cubic halves of side L: the ABOVE half (z > L) and the BELOW half (z < L), separated by an invisible wall (the 'door') at z = L. There are N atoms total, split evenly with N/2 starting in each half, all initialized at the same temperature T, moving freely and bouncing elastically off the outer walls. Particles interact with each other via a van der Waals potential, so their velocities are not fixed — they evolve from collisions and forces, not just free flight.

## Your goal
Maximize the absolute temperature difference |T_above - T_below| between the two halves, even though both halves begin at the same temperature. Temperature is proportional to the average kinetic energy of particles in each half. You win by using the door to sort fast (hot) particles into one half and slow (cold) particles into the other.

## Particle balance constraint
You MUST end the simulation with exactly N/2 particles on each side. Any temperature difference achieved with an unequal particle split is invalid. Before calling `finished`, verify via `get_system` that both halves contain equal particle counts. If the counts are unequal, reopen the door and wait for particles to redistribute, or selectively allow particles to cross until balance is restored.

## The door rules
- When the door is OPEN: particles pass freely between halves.
- When the door is CLOSED: particles cannot cross z = L and bounce back elastically.
- A particle's 'home' half is determined by which side it was on when the door last acted on it — closing the door traps each particle in whichever half it currently occupies.

## Strategy hints
The optimal agent watches individual particle velocities and positions, then:
1. Opens the door briefly to let a fast particle cross from BELOW to ABOVE (or a slow one from ABOVE to BELOW).
2. Closes the door immediately after to trap the temperature asymmetry.
3. Always swaps particles in pairs (one fast crossing up for every slow crossing down) to keep counts balanced.
A simpler but effective heuristic: if T_below > T_above, open the door so heat flows upward on average; once T_above > T_below, close the door to lock in the difference. Repeat, always reinforcing whichever half is already hotter. Track running counts throughout and correct any imbalance before finishing.

## Termination
When you are satisfied with the achieved temperature difference, call the `finished` tool to release control of the simulation. You do NOT need to reach a perfect outcome — stop when further improvement seems unlikely. Reminder: `finished` is only valid when n_above == n_below == N/2.
""".strip()

    @staticmethod
    def user_prompt(dyn: MolecularDynamics) -> str:
        return f"The current step index is {dyn.nsteps}."

    def serialize(self) -> str:
        return self.model_dump_json()


class SystemDebug(System):
    @staticmethod
    def system_prompt() -> str:
        return """
## Setup
You are controlling a molecular dynamics simulation. N atoms move inside two boxes of height L, bouncing elastically off the walls. The box is split into two halves by an invisible wall (the 'door') at z = L/2: the ABOVE half (z > L/2) and the BELOW half (z < L/2).

## Instructions
Verify that all tools work as described.

## Termination
When you are satisfied that all tools are working properly call the `finished` tool to release control of the simulation.
""".strip()

    @staticmethod
    def user_prompt(dyn: MolecularDynamics) -> str:
        return f"The current step index is {dyn.nsteps}."

    def serialize(self) -> str:
        return self.model_dump_json()
