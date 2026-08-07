"""Training loop sanity: reference fit, learning progress, fair ridge tuning."""

import numpy as np
import pytest

from descpot.descriptors import AcsfParams
from descpot.model import MorsePotential
from descpot.structures import rattle_batch
from descpot.training import (
    TrainSettings,
    evaluate_batches,
    fit_reference,
    make_bpnn,
    prepare_batches,
    train_potential,
)


def _synthetic_frames(n_batches=4, per_batch=5):
    """Frames labeled by a simple analytic energy so training can be verified."""
    frames = []
    for comp in ("Ti", "Zr", "Nb", "TiZrNb"):
        for b in range(n_batches):
            fs = rattle_batch(comp, 2, 0.08, per_batch, seed=b + 10, batch=b)
            for fr in fs:
                # composition-linear part + a smooth structural part
                mu = np.array([-4.0, -5.0, -6.0])
                spread = float(np.var(fr.positions - fr.positions.mean(axis=0)))
                fr.energy = float(fr.counts() @ mu + 0.02 * spread * fr.n_atoms)
                fr.forces = np.zeros_like(fr.positions)
            frames.extend(fs)
    return frames


def test_fit_reference_recovers_linear_part():
    frames = _synthetic_frames()
    mu = fit_reference(frames)
    assert mu == pytest.approx([-4.0, -5.0, -6.0], abs=0.2)


def test_training_reduces_energy_error():
    frames = _synthetic_frames()
    idx = list(range(len(frames)))
    train_idx, val_idx = idx[: int(0.8 * len(idx))], idx[int(0.8 * len(idx)) :]
    model = make_bpnn(hidden=(16,), seed=0)
    settings = TrainSettings(epochs=10, batch_size=8, lr=3e-3, force_weight=0.0, val_every=2)
    hist = train_potential(model, frames, train_idx, val_idx, settings)
    assert hist["loss_e"][-1] < hist["loss_e"][0]
    assert hist["best_epoch"] >= 0
    assert hist["n_parameters"] == model.n_parameters


def test_evaluate_batches_metric_units():
    frames = _synthetic_frames(n_batches=2, per_batch=3)
    model = MorsePotential()
    model.set_reference(fit_reference(frames))
    model.eval()
    batches = prepare_batches(frames, list(range(len(frames))), AcsfParams(), 6)
    metrics = evaluate_batches(model, batches)
    assert metrics["n_structures"] == len(frames)
    assert metrics["e_mae_mev_per_atom"] > 0
    assert len(metrics["e_pred_per_atom"]) == len(frames)
    # rmse >= mae always
    assert metrics["e_rmse_mev_per_atom"] >= metrics["e_mae_mev_per_atom"] - 1e-9
    assert metrics["f_rmse_mev_per_a"] >= metrics["f_mae_mev_per_a"] - 1e-9


def test_force_weight_changes_optimization():
    frames = _synthetic_frames(n_batches=2, per_batch=4)
    rng = np.random.default_rng(0)
    for fr in frames:
        fr.forces = rng.normal(0, 0.3, fr.positions.shape)
    idx = list(range(len(frames)))
    train_idx, val_idx = idx[:12], idx[12:]
    losses = {}
    for fw in (0.0, 1.0):
        model = make_bpnn(hidden=(8,), seed=1)
        settings = TrainSettings(epochs=3, batch_size=6, force_weight=fw, val_every=3)
        # the first 12 frames span only two of three elements, so the
        # rank-deficiency guard on the reference fit must fire
        with pytest.warns(UserWarning, match="rank-deficient"):
            hist = train_potential(model, frames, train_idx, val_idx, settings)
        losses[fw] = hist["loss_f"][-1]
    # with force_weight=0 the force loss is not even computed
    assert losses[0.0] == 0.0
    assert losses[1.0] > 0.0
