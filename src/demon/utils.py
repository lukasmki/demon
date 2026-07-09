import numpy as np
from ase import Atoms
from ase.md import thermalize_momenta

from demon.calculator import WalledVdW


def setup_system(
    N: int = 20,
    V: float = 8000,
    T: float = 5000,
    VA: bool = True,
    VR: bool = True,
    seed: int = 42,
) -> Atoms:
    np.random.seed(seed)
    L = pow(V, 1 / 3)
    N_per_side = int(pow(N, 1 / 3))
    min_dist = 0.5 * (L / N_per_side)

    above = make_box(N=N, L=L, T=T, min_dist=min_dist)
    below = make_box(N=N, L=L, T=T, min_dist=min_dist)
    above.translate([0, 0, L])
    atoms: Atoms = above + below
    atoms.set_cell([L, L, 2 * L, 90, 90, 90])
    atoms.set_pbc([False, False, False])

    atoms.calc = WalledVdW(
        epsilonij=0.0104,
        sigmaij=3.405,
        gamma=0.5 * 3.405,
        repulsive=VR,
        attractive=VA,
    )
    return atoms


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
    thermalize_momenta(atoms, temperature_K=T, exact_temperature=True)
    return atoms
