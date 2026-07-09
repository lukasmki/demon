from pathlib import Path
from typing import cast

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms, io


def plot_temperature_difference(traj: list[Atoms]):
    steps = np.arange(len(traj))
    T_a = np.array([atoms.info.get("T_a", float("nan")) for atoms in traj])
    T_b = np.array([atoms.info.get("T_b", float("nan")) for atoms in traj])
    door = [atoms.info.get("door", "open") for atoms in traj]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(steps, T_a, label="T above", color="tomato")
    ax1.plot(steps, T_b, label="T below", color="steelblue")
    ax1.set_ylabel("Temperature (K)")
    ax1.legend()

    delta_T = T_a - T_b
    ax2.plot(steps, delta_T, color="purple")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("ΔT = T_above − T_below (K)")
    ax2.set_xlabel("Step")

    # shade closed-door regions
    closed = np.array(door) == "closed"
    for ax in (ax1, ax2):
        for start, end in _runs(closed, steps):
            ax.axvspan(start, end, alpha=0.12, color="red", label="_")

    fig.suptitle("Maxwell's Demon — Temperature")
    fig.tight_layout()
    return fig


def plot_animation(traj: list[Atoms]) -> tuple:
    cell = traj[0].get_cell().array
    Lx, Lz = cell[0, 0], cell[2, 2]

    # pre-compute per-frame speeds for a consistent colour scale
    all_speeds = np.concatenate(
        [np.linalg.norm(atoms.get_velocities(), axis=1) for atoms in traj]
    )
    vmin, vmax = np.percentile(all_speeds, 5), np.percentile(all_speeds, 95)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Lz)
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("z (Å)")
    ax.set_aspect("equal")
    ax.set_facecolor("#f5f5f5")

    # door line
    (door_line,) = ax.plot(
        [0, Lx],
        [0.5 * Lz, 0.5 * Lz],
        color="green",
        linewidth=2,
        linestyle="--",
        zorder=3,
    )

    sc = ax.scatter(
        [],
        [],
        s=40,
        c=[],
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        edgecolors="none",
        zorder=4,
    )
    fig.colorbar(sc, ax=ax, label="Speed (Å/fs)")

    info_text = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=9, family="monospace"
    )
    temp_text = ax.text(
        0.98, 0.98, "", transform=ax.transAxes, va="top", ha="right", fontsize=9
    )

    def _update(frame):
        atoms = traj[frame]
        pos = atoms.get_positions()
        speeds = np.linalg.norm(atoms.get_velocities(), axis=1)
        sc.set_offsets(pos[:, [0, 2]])
        sc.set_array(speeds)

        door = atoms.info.get("door", "open")
        if door == "closed":
            door_line.set_color("red")
            door_line.set_linestyle("-")
            door_line.set_linewidth(3)
        else:
            door_line.set_color("green")
            door_line.set_linestyle("--")
            door_line.set_linewidth(2)

        info_text.set_text(f"Step {frame}  door: {door}")
        T_a = atoms.info.get("T_a", float("nan"))
        T_b = atoms.info.get("T_b", float("nan"))
        temp_text.set_text(f"T↑ {T_a:6.0f} K\nT↓ {T_b:6.0f} K\nΔT {T_a - T_b:+.0f} K")

        return sc, door_line, info_text, temp_text

    anim = animation.FuncAnimation(
        fig, _update, frames=len(traj), interval=50, blit=True
    )
    return fig, anim


def _runs(mask: np.ndarray, x: np.ndarray):
    """Yield (x_start, x_end) for each contiguous True run in mask."""
    in_run = False
    for i, val in enumerate(mask):
        if val and not in_run:
            start = x[i]
            in_run = True
        elif not val and in_run:
            yield start, x[i - 1]
            in_run = False
    if in_run:
        yield start, x[-1]


if __name__ == "__main__":
    root = Path(__file__).parent
    traj = cast(list[Atoms], io.read(root / "simulation.xyz", index=":"))

    fig_temp = plot_temperature_difference(traj)
    fig_temp.savefig(root / "temperature_difference.png", dpi=150, bbox_inches="tight")
    print("Saved temperature_difference.png")

    fig_anim, anim = plot_animation(traj)
    anim.save(root / "trajectory.gif", writer="pillow", fps=20)
    print("Saved trajectory.gif")

    plt.show()
