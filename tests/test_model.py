"""Model correctness: autograd forces vs finite differences, extensivity, IO."""

import numpy as np
import pytest
import torch

from descpot.data import Frame
from descpot.descriptors import AcsfParams, build_graph
from descpot.model import (
    BpnnPotential,
    LinearPotential,
    MorsePotential,
    load_potential,
    save_potential,
)
from descpot.structures import bcc_supercell


def _rattled(comp="TiZrNb", reps=2, seed=0, amp=0.1):
    rng = np.random.default_rng(seed)
    fr = bcc_supercell(comp, reps, seed=seed)
    fr.positions = fr.positions + rng.normal(0, amp, fr.positions.shape)
    return fr


def _make(kind):
    torch.manual_seed(0)
    if kind == "bpnn":
        model = BpnnPotential(AcsfParams(), hidden=(16,))
        # non-trivial weights so force check is meaningful
        for p in model.parameters():
            torch.nn.init.normal_(p, std=0.1)
    elif kind == "linear":
        model = LinearPotential(AcsfParams())
        torch.nn.init.normal_(model.weight, std=0.05)
    else:
        model = MorsePotential()
    model.set_reference(np.array([-5.0, -6.0, -7.0]))
    model.eval()
    return model


@pytest.mark.parametrize("kind", ["bpnn", "linear", "morse"])
def test_forces_match_finite_differences(kind):
    model = _make(kind)
    fr = _rattled(seed=11)
    graph = build_graph([fr], AcsfParams())
    _, forces = model.energy_forces(graph)
    forces = forces.detach().numpy()
    h = 1e-5
    rng = np.random.default_rng(0)
    for _ in range(6):
        a = int(rng.integers(fr.n_atoms))
        d = int(rng.integers(3))
        for sign, store in ((1, "p"), (-1, "m")):
            fr2 = Frame(fr.species.copy(), fr.positions.copy(), fr.cell.copy())
            fr2.positions[a, d] += sign * h
            g2 = build_graph([fr2], AcsfParams())
            with torch.no_grad():
                e = float(model(g2.positions, g2)[0])
            if store == "p":
                ep = e
            else:
                em = e
        fd = -(ep - em) / (2 * h)
        assert forces[a, d] == pytest.approx(fd, abs=1e-6, rel=1e-5)


@pytest.mark.parametrize("kind", ["bpnn", "linear", "morse"])
def test_energy_extensivity(kind):
    """Ideal elemental BCC: energy per atom identical for 2x2x2 and 3x3x3."""
    model = _make(kind)
    e = {}
    for reps in (2, 3):
        fr = bcc_supercell("Zr", reps, seed=0)
        graph = build_graph([fr], AcsfParams())
        with torch.no_grad():
            e[reps] = float(model(graph.positions, graph)[0]) / fr.n_atoms
    assert e[2] == pytest.approx(e[3], abs=1e-9)


@pytest.mark.parametrize("kind", ["bpnn", "linear", "morse"])
def test_batched_energies_match_single(kind):
    model = _make(kind)
    fra, frb = _rattled(seed=1), _rattled("ZrNb", seed=2)
    singles = []
    for fr in (fra, frb):
        g = build_graph([fr], AcsfParams())
        with torch.no_grad():
            singles.append(float(model(g.positions, g)[0]))
    g2 = build_graph([fra, frb], AcsfParams())
    with torch.no_grad():
        both = model(g2.positions, g2).numpy()
    assert np.allclose(both, singles, atol=1e-9)


@pytest.mark.parametrize("kind", ["bpnn", "linear", "morse"])
def test_save_load_roundtrip(kind, tmp_path):
    model = _make(kind)
    fr = _rattled(seed=4)
    graph = build_graph([fr], AcsfParams())
    with torch.no_grad():
        e0 = float(model(graph.positions, graph)[0])
    path = tmp_path / "model.pt"
    save_potential(model, path, extra={"note": "test"})
    loaded = load_potential(path)
    with torch.no_grad():
        e1 = float(loaded(graph.positions, graph)[0])
    assert e0 == pytest.approx(e1, abs=1e-12)


def test_scaler_dead_features_stay_bounded():
    """Features constant in training must not explode on unseen compositions."""
    import torch as _torch

    from descpot.descriptors import compute_descriptors
    from descpot.model import DescriptorScaler

    acsf = AcsfParams()
    train_fr = [_rattled("Nb", seed=k, amp=0.1) for k in range(3)]  # no Ti, no Zr
    g_train = build_graph(train_fr, acsf)
    d_train = compute_descriptors(g_train.positions, g_train, acsf)
    scaler = DescriptorScaler(acsf.n_features)
    scaler.fit(d_train, g_train.species)

    test_fr = [_rattled("TiZrNb", seed=9, amp=0.1)]  # all channels now active
    g_test = build_graph(test_fr, acsf)
    d_test = compute_descriptors(g_test.positions, g_test, acsf)
    x = scaler(d_test, g_test.species)
    assert _torch.isfinite(x).all()
    assert float(x.abs().max()) < 1e3


def test_parameter_counts_small():
    bpnn = BpnnPotential(AcsfParams(), hidden=(32, 32))
    assert bpnn.n_parameters < 20000
    linear = LinearPotential(AcsfParams())
    assert linear.n_parameters == 3 * AcsfParams().n_features + 3
    morse = MorsePotential()
    assert morse.n_parameters == 21
