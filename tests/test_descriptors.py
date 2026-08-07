"""ACSF invariance and consistency tests."""

import numpy as np
import torch

from descpot.data import Frame
from descpot.descriptors import AcsfParams, build_graph, compute_descriptors
from descpot.structures import bcc_supercell


def _rattled(comp="TiZrNb", reps=2, seed=0, amp=0.12):
    rng = np.random.default_rng(seed)
    fr = bcc_supercell(comp, reps, seed=seed)
    fr.positions = fr.positions + rng.normal(0, amp, fr.positions.shape)
    return fr


def _desc(fr, params=None):
    params = params or AcsfParams()
    graph = build_graph([fr], params)
    return compute_descriptors(graph.positions, graph, params).numpy()


def test_translation_invariance():
    fr = _rattled(seed=1)
    d0 = _desc(fr)
    fr2 = Frame(fr.species.copy(), fr.positions + np.array([1.7, -0.4, 2.9]), fr.cell.copy())
    assert np.allclose(d0, _desc(fr2), atol=1e-10)


def test_rotation_invariance():
    fr = _rattled(seed=2)
    d0 = _desc(fr)
    theta = 0.7
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    fr2 = Frame(fr.species.copy(), fr.positions @ rot.T, fr.cell @ rot.T)
    assert np.allclose(d0, _desc(fr2), atol=1e-9)


def test_permutation_equivariance():
    fr = _rattled(seed=3)
    d0 = _desc(fr)
    perm = np.random.default_rng(0).permutation(fr.n_atoms)
    fr2 = Frame(fr.species[perm], fr.positions[perm], fr.cell.copy())
    assert np.allclose(d0[perm], _desc(fr2), atol=1e-10)


def test_periodic_image_invariance():
    """Wrapping an atom by a lattice vector must not change any descriptor."""
    fr = _rattled(seed=4)
    d0 = _desc(fr)
    fr2 = Frame(fr.species.copy(), fr.positions.copy(), fr.cell.copy())
    fr2.positions[5] = fr2.positions[5] + fr2.cell[0] + fr2.cell[2]
    assert np.allclose(d0, _desc(fr2), atol=1e-10)


def test_supercell_size_consistency():
    """Ideal elemental BCC: every atom identical across supercell sizes."""
    params = AcsfParams()
    d2 = _desc(bcc_supercell("Nb", 2, seed=0), params)
    d3 = _desc(bcc_supercell("Nb", 3, seed=0), params)
    assert np.allclose(d2.std(axis=0), 0.0, atol=1e-10)
    assert np.allclose(d2[0], d3[0], atol=1e-10)


def test_batching_matches_single():
    params = AcsfParams()
    fr_a = _rattled(seed=5)
    fr_b = _rattled("TiNb", seed=6)
    da = _desc(fr_a, params)
    db = _desc(fr_b, params)
    graph = build_graph([fr_a, fr_b], params)
    d_all = compute_descriptors(graph.positions, graph, params).numpy()
    assert np.allclose(d_all[: fr_a.n_atoms], da, atol=1e-12)
    assert np.allclose(d_all[fr_a.n_atoms :], db, atol=1e-12)


def test_descriptors_differentiable():
    params = AcsfParams()
    fr = _rattled(seed=7)
    graph = build_graph([fr], params)
    pos = graph.positions.clone().requires_grad_(True)
    desc = compute_descriptors(pos, graph, params)
    desc.sum().backward()
    assert pos.grad is not None
    assert torch.isfinite(pos.grad).all()
    assert float(pos.grad.abs().max()) > 0
