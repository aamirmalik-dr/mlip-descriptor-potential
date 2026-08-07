"""Equation-of-state scans and Birch-Murnaghan fits.

Compares the student potential against the teacher on ideal BCC cells, and
against external DFT anchors (Materials Project or literature) for lattice
constants and bulk moduli.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import curve_fit

from descpot.data import Frame
from descpot.descriptors import build_graph
from descpot.model import PotentialBase
from descpot.structures import bcc_supercell, vegard_a

EV_PER_A3_TO_GPA = 160.2176634


def scaled_frame(base: Frame, scale: float) -> Frame:
    """Isotropically rescale a frame's cell and positions."""
    return Frame(
        species=base.species.copy(),
        positions=base.positions * scale,
        cell=base.cell * scale,
        group=base.group,
        composition=base.composition,
    )


def ev_curve_model(
    model: PotentialBase, composition: str, reps: int, scales: np.ndarray, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Energy-volume curve of the student potential on ideal BCC cells.

    Returns:
        (volumes_per_atom, energies_per_atom) arrays in A^3 and eV.
    """
    base = bcc_supercell(composition, reps, seed=seed)
    vols, energies = [], []
    for s in scales:
        fr = scaled_frame(base, float(s))
        graph = build_graph([fr], model.acsf if hasattr(model, "acsf") else _default_acsf())
        with torch.no_grad():
            e = float(model(graph.positions, graph)[0])
        vols.append(abs(np.linalg.det(fr.cell)) / fr.n_atoms)
        energies.append(e / fr.n_atoms)
    return np.array(vols), np.array(energies)


def _default_acsf():
    from descpot.descriptors import AcsfParams

    return AcsfParams()


def ev_curve_teacher(
    calc, composition: str, reps: int, scales: np.ndarray, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Energy-volume curve of the teacher on the identical cells."""
    from descpot.teacher import frame_to_atoms

    base = bcc_supercell(composition, reps, seed=seed)
    vols, energies = [], []
    for s in scales:
        fr = scaled_frame(base, float(s))
        atoms = frame_to_atoms(fr)
        atoms.calc = calc
        energies.append(float(atoms.get_potential_energy()) / fr.n_atoms)
        vols.append(abs(np.linalg.det(fr.cell)) / fr.n_atoms)
    return np.array(vols), np.array(energies)


def birch_murnaghan(v: np.ndarray, e0: float, v0: float, b0: float, b0p: float) -> np.ndarray:
    """Third-order Birch-Murnaghan energy-volume relation."""
    eta = (v0 / v) ** (2.0 / 3.0)
    return e0 + 9.0 * v0 * b0 / 16.0 * (
        (eta - 1.0) ** 3 * b0p + (eta - 1.0) ** 2 * (6.0 - 4.0 * eta)
    )


def fit_eos(volumes: np.ndarray, energies: np.ndarray) -> dict:
    """Fit Birch-Murnaghan; returns a0 (BCC), V0, B0 in GPa, B0', E0.

    Args:
        volumes: Volumes per atom, A^3.
        energies: Energies per atom, eV.
    """
    v0_guess = volumes[np.argmin(energies)]
    p0 = [float(energies.min()), float(v0_guess), 0.6, 4.0]
    popt, _ = curve_fit(birch_murnaghan, volumes, energies, p0=p0, maxfev=20000)
    e0, v0, b0, b0p = popt
    return {
        "e0_ev_per_atom": float(e0),
        "v0_a3_per_atom": float(v0),
        "a0_bcc_a": float((2.0 * v0) ** (1.0 / 3.0)),
        "b0_gpa": float(b0 * EV_PER_A3_TO_GPA),
        "b0_prime": float(b0p),
    }


def default_scales(composition: str, n: int = 13, span: float = 0.06) -> np.ndarray:
    """Symmetric scale grid around the Vegard-rule lattice constant."""
    del composition
    return np.linspace(1.0 - span, 1.0 + span, n)


def vegard_reference(composition: str) -> float:
    """Expose the generation-time Vegard lattice constant for reporting."""
    return vegard_a(composition)
