"""Neighbor-list correctness against ASE, including strained cells."""

import numpy as np
import pytest
from ase.neighborlist import neighbor_list as ase_neighbor_list

from descpot.neighbors import neighbor_pairs, restrict_pairs, triplets_from_pairs
from descpot.structures import bcc_supercell
from descpot.teacher import frame_to_atoms


def _pair_set(i, j, shift):
    return {(int(a), int(b), tuple(int(x) for x in s)) for a, b, s in zip(i, j, shift)}


@pytest.mark.parametrize("comp,reps", [("Nb", 2), ("TiZrNb", 2), ("TiZr", 3)])
def test_matches_ase_ideal(comp, reps):
    fr = bcc_supercell(comp, reps, seed=3)
    pairs = neighbor_pairs(fr.positions, fr.cell, cutoff=5.5)
    ai, aj, aS = ase_neighbor_list("ijS", frame_to_atoms(fr), 5.5)
    assert _pair_set(pairs.i, pairs.j, pairs.shift) == _pair_set(ai, aj, aS)


def test_matches_ase_strained_cell():
    fr = bcc_supercell("TiZrNb", 2, seed=1)
    rng = np.random.default_rng(0)
    strain = np.eye(3) + rng.uniform(-0.05, 0.05, (3, 3))
    frac = fr.positions @ np.linalg.inv(fr.cell)
    fr.cell = fr.cell @ strain.T
    fr.positions = frac @ fr.cell + rng.normal(0, 0.1, fr.positions.shape)
    pairs = neighbor_pairs(fr.positions, fr.cell, cutoff=5.0)
    ai, aj, aS = ase_neighbor_list("ijS", frame_to_atoms(fr), 5.0)
    assert _pair_set(pairs.i, pairs.j, pairs.shift) == _pair_set(ai, aj, aS)


def test_directed_pairs_symmetric():
    fr = bcc_supercell("Ti", 2, seed=0)
    pairs = neighbor_pairs(fr.positions, fr.cell, cutoff=4.0)
    fwd = _pair_set(pairs.i, pairs.j, pairs.shift)
    rev = {(j, i, tuple(-np.array(s))) for i, j, s in fwd}
    assert fwd == rev


def test_restrict_pairs_distances():
    fr = bcc_supercell("ZrNb", 2, seed=5)
    pairs = neighbor_pairs(fr.positions, fr.cell, cutoff=5.5)
    keep = restrict_pairs(pairs, fr.positions, fr.cell, cutoff=4.0)
    vec = fr.positions[pairs.j] + pairs.shift @ fr.cell - fr.positions[pairs.i]
    dist = np.linalg.norm(vec, axis=1)
    assert np.all(dist[keep] < 4.0)
    outside = np.setdiff1d(np.arange(len(pairs.i)), keep)
    assert np.all(dist[outside] >= 4.0)


def test_triplet_counts():
    fr = bcc_supercell("Nb", 2, seed=0)
    pairs = neighbor_pairs(fr.positions, fr.cell, cutoff=3.0)
    tri = triplets_from_pairs(pairs)
    # ideal BCC first shell within 3.0 A: 8 neighbors -> C(8,2)=28 per center
    counts = np.bincount(tri.center, minlength=fr.n_atoms)
    assert np.all(counts == 28)
    # each triplet's legs share the center
    assert np.all(pairs.i[tri.pair_ij] == tri.center)
    assert np.all(pairs.i[tri.pair_ik] == tri.center)
    # unordered: no duplicated (ij, ik) swap
    seen = set(zip(tri.pair_ij, tri.pair_ik))
    assert len(seen) == len(tri.pair_ij)
    assert all((b, a) not in seen for a, b in seen)
