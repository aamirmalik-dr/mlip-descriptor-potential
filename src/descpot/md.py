"""Molecular dynamics with the student potential: NVE stability check.

Velocity-Verlet integration implemented from scratch in internal units
(eV, Angstrom, amu). The headline diagnostic is total-energy drift per atom
per picosecond, the standard smoke test for force consistency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from descpot.data import Frame
from descpot.descriptors import build_graph
from descpot.model import PotentialBase

MASSES_AMU = np.array([47.867, 91.224, 92.906])  # Ti, Zr, Nb
# 1 internal time unit = A * sqrt(amu / eV) = 10.180506 fs
FS_PER_INTERNAL = 10.180506
KB_EV = 8.617333262e-5


@dataclass
class NveResult:
    """Trace and drift diagnostics of one NVE run."""

    times_ps: np.ndarray
    e_pot: np.ndarray
    e_kin: np.ndarray
    e_tot: np.ndarray
    drift_mev_per_atom_per_ps: float
    temperature_mean_k: float


def _forces(model: PotentialBase, frame: Frame) -> tuple[float, np.ndarray]:
    graph = build_graph([frame], model.acsf if hasattr(model, "acsf") else _acsf())
    energies, forces = model.energy_forces(graph)
    return float(energies.detach()[0]), forces.detach().numpy()


def _acsf():
    from descpot.descriptors import AcsfParams

    return AcsfParams()


def run_nve(
    model: PotentialBase,
    frame: Frame,
    temperature_k: float = 600.0,
    timestep_fs: float = 2.0,
    n_steps: int = 1000,
    seed: int = 0,
    sample_every: int = 5,
) -> NveResult:
    """Integrate NVE dynamics and measure total-energy drift.

    Velocities are drawn from Maxwell-Boltzmann at temperature_k with the
    center-of-mass motion removed, so the observed kinetic temperature settles
    near half the target after equipartition with potential modes.

    Args:
        model: Trained potential (eval mode).
        frame: Starting structure.
        temperature_k: Initial velocity temperature.
        timestep_fs: Timestep in femtoseconds.
        n_steps: Number of Verlet steps.
        seed: Velocity RNG seed.
        sample_every: Recording stride.

    Returns:
        NveResult with energy traces (eV/atom) and the linear drift rate.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = frame.n_atoms
    masses = MASSES_AMU[frame.species][:, None]
    dt = timestep_fs / FS_PER_INTERNAL

    vel = rng.normal(0.0, 1.0, (n, 3)) * np.sqrt(KB_EV * temperature_k / masses)
    vel -= (vel * masses).sum(axis=0) / masses.sum()

    fr = Frame(
        species=frame.species.copy(),
        positions=frame.positions.copy(),
        cell=frame.cell.copy(),
        composition=frame.composition,
    )
    e_pot, forces = _forces(model, fr)
    times, pots, kins = [], [], []
    for step in range(n_steps + 1):
        if step % sample_every == 0:
            ke = float(0.5 * (masses * vel * vel).sum())
            times.append(step * timestep_fs / 1000.0)
            pots.append(e_pot / n)
            kins.append(ke / n)
        if step == n_steps:
            break
        vel = vel + 0.5 * dt * forces / masses
        fr.positions = fr.positions + dt * vel
        e_pot, forces = _forces(model, fr)
        vel = vel + 0.5 * dt * forces / masses

    times_arr = np.array(times)
    pots_arr = np.array(pots)
    kins_arr = np.array(kins)
    etot = pots_arr + kins_arr
    slope = float(np.polyfit(times_arr, etot, 1)[0]) if len(times_arr) > 2 else 0.0
    mean_temp = float(2.0 * kins_arr.mean() / (3.0 * KB_EV))
    return NveResult(
        times_ps=times_arr,
        e_pot=pots_arr,
        e_kin=kins_arr,
        e_tot=etot,
        drift_mev_per_atom_per_ps=slope * 1000.0,
        temperature_mean_k=mean_temp,
    )
