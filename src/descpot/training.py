"""Training loop: weighted energy-plus-force loss, shared by all three models.

The BPNN, the linear ridge-style model, and the Morse baseline all train with
the same optimizer, loss, batches, and epoch budget, so comparisons isolate
the representation rather than the tuning effort.
"""

from __future__ import annotations

import copy
import time
import warnings
from dataclasses import asdict, dataclass

import numpy as np
import torch

from descpot.data import Frame
from descpot.descriptors import Graph, build_graph, compute_descriptors
from descpot.model import BpnnPotential, LinearPotential, PotentialBase


@dataclass
class TrainSettings:
    """Hyperparameters of one training run.

    force_weight multiplies the force MSE (eV^2/A^2) against the per-atom
    energy MSE (eV^2/atom^2); energy weight is fixed at 1.
    """

    epochs: int = 60
    batch_size: int = 24
    lr: float = 3e-3
    weight_decay: float = 0.0
    force_weight: float = 0.1
    val_every: int = 5
    lr_min_factor: float = 0.05
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def fit_reference(frames: list[Frame]) -> np.ndarray:
    """Least-squares per-element reference energies from total energies.

    Warns when the composition matrix is rank-deficient: an element absent
    from (or perfectly correlated within) the fitting frames gets an
    arbitrary reference energy, which breaks predictions for compositions
    containing it.
    """
    counts = np.stack([fr.counts() for fr in frames]).astype(float)
    energies = np.array([fr.energy for fr in frames])
    if np.linalg.matrix_rank(counts) < counts.shape[1]:
        warnings.warn(
            "composition matrix is rank-deficient; per-element reference "
            "energies are underdetermined for at least one element",
            stacklevel=2,
        )
    mu, *_ = np.linalg.lstsq(counts, energies, rcond=None)
    return mu


def _batch_indices(n: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    perm = rng.permutation(n)
    return [perm[k : k + batch_size] for k in range(0, n, batch_size)]


@dataclass
class PreparedBatch:
    graph: Graph
    energy: torch.Tensor  # (S,) eV
    forces: torch.Tensor  # (N, 3) eV/A
    n_atoms: torch.Tensor  # (S,)


def prepare_batches(
    frames: list[Frame], indices: list[int], acsf, batch_size: int, seed: int = 0
) -> list[PreparedBatch]:
    """Pack frames into fixed batches with prebuilt neighbor graphs."""
    rng = np.random.default_rng(seed)
    batches = []
    for chunk in _batch_indices(len(indices), batch_size, rng):
        subset = [frames[indices[c]] for c in chunk]
        graph = build_graph(subset, acsf)
        batches.append(
            PreparedBatch(
                graph=graph,
                energy=torch.tensor([fr.energy for fr in subset], dtype=torch.float64),
                forces=torch.tensor(
                    np.concatenate([fr.forces for fr in subset]), dtype=torch.float64
                ),
                n_atoms=graph.n_atoms_per_struct.to(torch.float64),
            )
        )
    return batches


def fit_scaler(model: PotentialBase, batches: list[PreparedBatch]) -> None:
    """Fit per-element descriptor standardization on the training batches."""
    if not hasattr(model, "scaler"):
        return
    descs, specs = [], []
    with torch.no_grad():
        for b in batches:
            descs.append(compute_descriptors(b.graph.positions, b.graph, model.acsf))
            specs.append(b.graph.species)
    model.scaler.fit(torch.cat(descs), torch.cat(specs))


def _batch_loss(
    model: PotentialBase, batch: PreparedBatch, force_weight: float
) -> tuple[torch.Tensor, float, float]:
    if force_weight > 0:
        pos = batch.graph.positions.detach().clone().requires_grad_(True)
        energies = model(pos, batch.graph)
        (grad,) = torch.autograd.grad(energies.sum(), pos, create_graph=model.training)
        forces = -grad
        loss_f = ((forces - batch.forces) ** 2).mean()
    else:
        energies = model(batch.graph.positions, batch.graph)
        loss_f = torch.zeros((), dtype=torch.float64)
    loss_e = (((energies - batch.energy) / batch.n_atoms) ** 2).mean()
    loss = loss_e + force_weight * loss_f
    return loss, float(loss_e.detach()), float(loss_f.detach())


def evaluate_batches(model: PotentialBase, batches: list[PreparedBatch]) -> dict:
    """Energy and force errors plus parity arrays over prepared batches."""
    model.eval()
    e_pred, e_true, na = [], [], []
    f_pred, f_true = [], []
    for b in batches:
        energies, forces = model.energy_forces(b.graph)
        e_pred.append(energies.detach().numpy())
        e_true.append(b.energy.numpy())
        na.append(b.n_atoms.numpy())
        f_pred.append(forces.detach().numpy())
        f_true.append(b.forces.numpy())
    e_pred = np.concatenate(e_pred)
    e_true = np.concatenate(e_true)
    na = np.concatenate(na)
    f_pred = np.concatenate(f_pred)
    f_true = np.concatenate(f_true)
    de = (e_pred - e_true) / na
    df = f_pred - f_true
    return {
        "e_mae_mev_per_atom": float(np.abs(de).mean() * 1000),
        "e_rmse_mev_per_atom": float(np.sqrt((de**2).mean()) * 1000),
        "f_mae_mev_per_a": float(np.abs(df).mean() * 1000),
        "f_rmse_mev_per_a": float(np.sqrt((df**2).mean()) * 1000),
        "n_structures": int(len(e_true)),
        "e_pred_per_atom": (e_pred / na).tolist(),
        "e_true_per_atom": (e_true / na).tolist(),
        "f_pred_sample": f_pred[:, 0].tolist(),
        "f_true_sample": f_true[:, 0].tolist(),
    }


def train_potential(
    model: PotentialBase,
    frames: list[Frame],
    train_idx: list[int],
    val_idx: list[int],
    settings: TrainSettings,
) -> dict:
    """Train any potential with the shared weighted energy+force loss.

    Fits the composition-linear reference and (where applicable) the feature
    scaler on the training split only, then optimizes with Adam and cosine
    learning-rate decay, keeping the best validation state.

    Returns:
        History dict with per-epoch losses, validation metrics, and timing.
    """
    torch.manual_seed(settings.seed)
    np.random.seed(settings.seed)
    mu = fit_reference([frames[i] for i in train_idx])
    model.set_reference(mu)
    acsf = getattr(model, "acsf", None)
    from descpot.descriptors import AcsfParams

    graph_acsf = acsf if acsf is not None else AcsfParams()
    train_batches = prepare_batches(frames, train_idx, graph_acsf, settings.batch_size, seed=1)
    val_batches = prepare_batches(frames, val_idx, graph_acsf, settings.batch_size, seed=2)
    fit_scaler(model, train_batches)

    opt = torch.optim.Adam(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=settings.epochs, eta_min=settings.lr * settings.lr_min_factor
    )
    rng = np.random.default_rng(settings.seed + 7)
    history: dict = {
        "epoch": [],
        "loss_e": [],
        "loss_f": [],
        "val": [],
        "settings": settings.to_dict(),
    }
    best = {"score": np.inf, "state": None, "epoch": -1}
    start = time.perf_counter()
    for epoch in range(settings.epochs):
        model.train()
        order = rng.permutation(len(train_batches))
        sum_e, sum_f = 0.0, 0.0
        for k in order:
            opt.zero_grad()
            loss, le, lf = _batch_loss(model, train_batches[k], settings.force_weight)
            loss.backward()
            opt.step()
            sum_e += le
            sum_f += lf
        sched.step()
        history["epoch"].append(epoch)
        history["loss_e"].append(sum_e / len(train_batches))
        history["loss_f"].append(sum_f / len(train_batches))
        if (epoch + 1) % settings.val_every == 0 or epoch == settings.epochs - 1:
            metrics = evaluate_batches(model, val_batches)
            score = metrics["e_mae_mev_per_atom"] + 0.1 * metrics["f_mae_mev_per_a"]
            history["val"].append(
                {
                    "epoch": epoch,
                    "e_mae_mev_per_atom": metrics["e_mae_mev_per_atom"],
                    "f_mae_mev_per_a": metrics["f_mae_mev_per_a"],
                    "score": score,
                }
            )
            if score < best["score"]:
                best = {
                    "score": score,
                    "state": copy.deepcopy(model.state_dict()),
                    "epoch": epoch,
                }
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()
    history["train_seconds"] = time.perf_counter() - start
    history["best_epoch"] = best["epoch"]
    history["n_parameters"] = model.n_parameters
    return history


def tune_linear_ridge(
    frames: list[Frame],
    train_idx: list[int],
    val_idx: list[int],
    settings: TrainSettings,
    acsf,
    alphas: tuple[float, ...] = (1e-6, 1e-4, 1e-2),
) -> tuple[LinearPotential, dict]:
    """Fair tuning for the linear baseline: sweep ridge strength on validation.

    Candidate alphas train with the identical loop at a third of the epoch
    budget (tuning cost, extra to the baseline's favor); the winner then
    retrains at the full budget, so the returned model's budget matches the
    BPNN's exactly.
    """
    tune_epochs = max(10, settings.epochs // 3)
    best_alpha, best_score = alphas[0], np.inf
    trials = []
    for alpha in alphas:
        model = LinearPotential(acsf)
        s = TrainSettings(**{**settings.to_dict(), "weight_decay": alpha, "epochs": tune_epochs})
        hist = train_potential(model, frames, train_idx, val_idx, s)
        score = min(v["score"] for v in hist["val"])
        trials.append({"alpha": alpha, "val_score_at_tune_epochs": score})
        if score < best_score:
            best_alpha, best_score = alpha, score
    final = LinearPotential(acsf)
    s = TrainSettings(**{**settings.to_dict(), "weight_decay": best_alpha})
    hist = train_potential(final, frames, train_idx, val_idx, s)
    hist["ridge_trials"] = trials
    hist["ridge_alpha"] = best_alpha
    return final, hist


def make_bpnn(acsf=None, hidden=(32, 32), seed: int = 0) -> BpnnPotential:
    """Convenience constructor with seeded initialization."""
    from descpot.descriptors import AcsfParams

    torch.manual_seed(seed)
    return BpnnPotential(acsf or AcsfParams(), tuple(hidden))
