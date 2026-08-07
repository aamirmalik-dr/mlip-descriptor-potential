"""Frame IO and leakage-safe split integrity."""

import numpy as np
import pytest

from descpot.data import group_split, load_frames, random_frame_split, save_frames
from descpot.structures import COMPOSITIONS, bcc_supercell, rattle_batch, strain_batch


def _labeled_frames():
    frames = []
    rng = np.random.default_rng(0)
    for comp in ("Ti", "TiZr", "TiZrNb", "Ti2ZrNb"):
        for batch in range(3):
            fs = rattle_batch(comp, 2, 0.1, 4, seed=batch, batch=batch)
            for fr in fs:
                fr.energy = float(rng.normal(-100, 1))
                fr.forces = rng.normal(0, 0.5, fr.positions.shape)
            frames.extend(fs)
    return frames


def test_extxyz_roundtrip(tmp_path):
    frames = _labeled_frames()[:6]
    path = tmp_path / "frames.extxyz"
    save_frames(frames, path)
    loaded = load_frames(path)
    assert len(loaded) == len(frames)
    for a, b in zip(frames, loaded):
        assert np.array_equal(a.species, b.species)
        assert np.allclose(a.positions, b.positions, atol=1e-6)
        assert np.allclose(a.cell, b.cell, atol=1e-6)
        assert a.energy == pytest.approx(b.energy, abs=1e-6)
        assert np.allclose(a.forces, b.forces, atol=1e-6)
        assert a.group == b.group and a.composition == b.composition


def test_group_split_no_group_leakage():
    frames = _labeled_frames()
    split = group_split(frames, holdout_compositions=("Ti2ZrNb",), seed=0)
    g_train = {frames[i].group for i in split.train}
    g_val = {frames[i].group for i in split.val}
    g_test = {frames[i].group for i in split.test}
    assert not (g_train & g_test) and not (g_train & g_val) and not (g_val & g_test)
    assert all(frames[i].composition == "Ti2ZrNb" for i in split.transfer)
    assert all(
        frames[i].composition != "Ti2ZrNb"
        for part in (split.train, split.val, split.test)
        for i in part
    )
    n = len(split.train) + len(split.val) + len(split.test) + len(split.transfer)
    assert n == len(frames)


def test_group_split_stratified_by_kind():
    """Every batch type must appear in the held-out test set."""
    from descpot.data import group_kind

    frames = []
    rng = np.random.default_rng(1)
    for comp in ("Ti", "Zr", "Nb", "TiZrNb"):
        for amp, batch in ((0.05, 0), (0.18, 1)):
            fs = rattle_batch(comp, 2, amp, 4, seed=batch, batch=batch)
            fs += strain_batch(comp, 2, 4, seed=7, batch=0)
            for fr in fs:
                fr.energy = float(rng.normal(-100, 1))
                fr.forces = rng.normal(0, 0.5, fr.positions.shape)
            frames.extend(fs)
    for seed in (0, 1, 2):
        split = group_split(frames, seed=seed)
        test_kinds = {group_kind(frames[i].group, frames[i].composition) for i in split.test}
        all_kinds = {group_kind(fr.group, fr.composition) for fr in frames}
        assert test_kinds == all_kinds


def test_random_split_mixes_groups():
    frames = _labeled_frames()
    split = random_frame_split(frames, holdout_compositions=("Ti2ZrNb",), seed=0)
    g_train = {frames[i].group for i in split.train}
    g_test = {frames[i].group for i in split.test}
    # the point of the control: sibling frames from one batch land on both sides
    assert g_train & g_test


def test_generation_batches_are_tagged():
    fs = rattle_batch("TiZrNb", 2, 0.05, 3, seed=0, batch=7)
    assert all(fr.group == "TiZrNb_rattle05_b7" for fr in fs)
    fs = strain_batch("ZrNb", 2, 4, seed=0, batch=1)
    assert all(fr.group.startswith("ZrNb_strain_") for fr in fs)
    assert all(fr.composition == "ZrNb" for fr in fs)


def test_composition_counts():
    for comp, frac in COMPOSITIONS.items():
        fr = bcc_supercell(comp, 2, seed=0)
        counts = fr.counts()
        assert counts.sum() == 16
        for c, f in zip(counts, frac):
            assert abs(c - f * 16) <= 1
