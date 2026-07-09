from typing import cast

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator


class WalledVdW(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        epsilonij: float,
        sigmaij: float,
        repulsive: bool = True,
        attractive: bool = True,
        gamma: float = 0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.Aij = 4.0 * epsilonij * sigmaij**12
        self.Bij = 4.0 * epsilonij * sigmaij**6

        self.attractive = attractive
        self.repulsive = repulsive

    def calculate(self, atoms=None, properties=["energy", "forces"], system_changes=[]):
        super().calculate(atoms, properties, system_changes)
        atoms = cast(Atoms, self.atoms)
        pos = atoms.get_positions()

        # get atoms in above and below subsystems
        door_state = atoms.info.get("door", "closed")
        if door_state == "open":
            mask = np.ones((len(atoms), len(atoms)), dtype=bool)
        else:
            cell = atoms.get_cell()
            above = pos[:, 2] > 0.5 * cell[2, 2]
            below = ~above
            mask = np.outer(above, above) | np.outer(below, below)
        np.fill_diagonal(mask, False)

        r = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, 3)
        r2 = np.sum(r**2, axis=-1)  # (N, N)
        np.fill_diagonal(r2, np.inf)  # exclude self-interactions

        # shielded interaction to prevent divergence
        ir2 = np.where(mask, 1.0 / (r2 + self.gamma**2), 0.0)
        ir6 = ir2**3
        ir12 = ir6**2

        # E = A/r^12 - B/r^6
        pair_energy = np.zeros_like(ir6)
        if self.attractive:
            pair_energy -= self.Bij * ir6
        if self.repulsive:
            pair_energy += self.Aij * ir12
        energy = 0.5 * np.sum(pair_energy)

        # F = (12A/r^13 - 6B/r^7) * r_hat
        scalar_forces = np.zeros_like(ir6)
        if self.attractive:
            scalar_forces -= 6.0 * self.Bij * ir6
        if self.repulsive:
            scalar_forces += 12.0 * self.Aij * ir12
        scalar_forces *= ir2
        forces = np.sum(scalar_forces[:, :, np.newaxis] * r, axis=1)  # (N, 3)

        self.results = {
            "energy": energy,
            "forces": forces,
        }
