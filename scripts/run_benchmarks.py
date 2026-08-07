"""Run the config-driven, fixed-seed benchmark suite.

Benchmarks (each writes results/<name>.json):
    main            BPNN vs tuned linear vs Morse at matched data and budget
    split_gap       group split vs the optimistic random-frame split
    learning_curve  error vs training-set size for all three models
    force_weight    force-loss weight sweep for the BPNN
    eos             E-V curves and Birch-Murnaghan fits vs teacher and anchors
    nve             NVE energy-drift stability check

Usage:
    python scripts/run_benchmarks.py [--only main,eos] [--data data/full/dataset.extxyz]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from descpot.data import group_split, load_frames, random_frame_split
from descpot.descriptors import AcsfParams
from descpot.eos import default_scales, ev_curve_model, ev_curve_teacher, fit_eos
from descpot.md import run_nve
from descpot.model import MorsePotential, load_potential, save_potential
from descpot.structures import bcc_supercell
from descpot.training import (
    TrainSettings,
    evaluate_batches,
    make_bpnn,
    prepare_batches,
    train_potential,
    tune_linear_ridge,
)

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"


def _cfg(name: str) -> dict:
    return yaml.safe_load((REPO / "configs" / f"{name}.yaml").read_text())


def _settings(cfg: dict) -> TrainSettings:
    return TrainSettings(**cfg["train"])


def _acsf(cfg: dict) -> AcsfParams:
    return AcsfParams.from_dict(cfg["acsf"]) if "acsf" in cfg else AcsfParams()


def _evaluate_all(model, frames, split, acsf, batch_size):
    out = {}
    for part in ("test", "transfer"):
        idx = getattr(split, part)
        if idx:
            batches = prepare_batches(frames, idx, acsf, batch_size, seed=99)
            out[part] = evaluate_batches(model, batches)
    return out


def bench_main(frames, cfg) -> dict:
    acsf = _acsf(cfg)
    settings = _settings(cfg)
    split = group_split(
        frames,
        holdout_compositions=tuple(cfg["holdout_compositions"]),
        seed=cfg["split_seed"],
    )
    result = {
        "split": {k: len(getattr(split, k)) for k in ("train", "val", "test", "transfer")},
        "n_features": acsf.n_features,
        "models": {},
    }

    bpnn = make_bpnn(acsf, hidden=tuple(cfg["hidden"]), seed=settings.seed)
    hist = train_potential(bpnn, frames, split.train, split.val, settings)
    result["models"]["bpnn"] = {
        "history": {k: hist[k] for k in ("loss_e", "loss_f", "val", "train_seconds", "best_epoch")},
        "n_parameters": hist["n_parameters"],
        "eval": _evaluate_all(bpnn, frames, split, acsf, settings.batch_size),
    }
    save_potential(
        bpnn,
        REPO / "models" / "bpnn_main.pt",
        extra={"settings": settings.to_dict(), "split_seed": cfg["split_seed"]},
    )

    linear, lin_hist = tune_linear_ridge(
        frames, split.train, split.val, settings, acsf, alphas=tuple(cfg["ridge_alphas"])
    )
    result["models"]["linear"] = {
        "ridge_trials": lin_hist["ridge_trials"],
        "train_seconds": lin_hist["train_seconds"],
        "n_parameters": lin_hist["n_parameters"],
        "eval": _evaluate_all(linear, frames, split, acsf, settings.batch_size),
    }
    save_potential(linear, REPO / "models" / "linear_main.pt")

    morse = MorsePotential(cutoff=acsf.rc_radial)
    morse_hist = train_potential(morse, frames, split.train, split.val, settings)
    result["models"]["morse"] = {
        "train_seconds": morse_hist["train_seconds"],
        "n_parameters": morse_hist["n_parameters"],
        "eval": _evaluate_all(morse, frames, split, acsf, settings.batch_size),
    }
    save_potential(morse, REPO / "models" / "morse_main.pt")
    return result


def bench_split_gap(frames, cfg) -> dict:
    """Group split vs the optimistic random-frame split, repeated over seeds.

    Each protocol trains the identical BPNN with the identical budget and is
    scored on its own test set; multiple split seeds separate the protocol
    effect from test-set-difficulty noise.
    """
    acsf = _acsf(cfg)
    settings = _settings(cfg)
    holdout = tuple(cfg["holdout_compositions"])
    out: dict = {"seeds": cfg["split_seeds"], "group": [], "random_frame": []}
    for split_seed in cfg["split_seeds"]:
        for key, splitter in (("group", group_split), ("random_frame", random_frame_split)):
            split = splitter(frames, holdout_compositions=holdout, seed=split_seed)
            bpnn = make_bpnn(acsf, hidden=tuple(cfg["hidden"]), seed=settings.seed)
            train_potential(bpnn, frames, split.train, split.val, settings)
            batches = prepare_batches(frames, split.test, acsf, settings.batch_size, seed=99)
            m = evaluate_batches(bpnn, batches)
            out[key].append(
                {
                    "split_seed": split_seed,
                    "n_test": m["n_structures"],
                    "e_mae_mev_per_atom": m["e_mae_mev_per_atom"],
                    "f_mae_mev_per_a": m["f_mae_mev_per_a"],
                }
            )
    for key in ("group", "random_frame"):
        es = [r["e_mae_mev_per_atom"] for r in out[key]]
        fs = [r["f_mae_mev_per_a"] for r in out[key]]
        out[f"{key}_mean"] = {
            "e_mae_mev_per_atom": float(np.mean(es)),
            "e_mae_std": float(np.std(es)),
            "f_mae_mev_per_a": float(np.mean(fs)),
            "f_mae_std": float(np.std(fs)),
        }
    out["note"] = (
        "the random-frame protocol is an optimistic control only; sibling frames "
        "from one generation batch appear on both sides of that split"
    )
    return out


def bench_learning_curve(frames, cfg) -> dict:
    acsf = _acsf(cfg)
    settings = _settings(cfg)
    split = group_split(
        frames,
        holdout_compositions=tuple(cfg["holdout_compositions"]),
        seed=cfg["split_seed"],
    )
    rng = np.random.default_rng(cfg["subset_seed"])
    order = rng.permutation(len(split.train))
    sizes = [s for s in cfg["sizes"] if s <= len(split.train)]
    if sizes[-1] != len(split.train):
        sizes.append(len(split.train))
    curves: dict = {"sizes": sizes, "models": {"bpnn": [], "linear": [], "morse": []}}
    for size in sizes:
        sub = [split.train[i] for i in order[:size]]
        bpnn = make_bpnn(acsf, hidden=tuple(cfg["hidden"]), seed=settings.seed)
        train_potential(bpnn, frames, sub, split.val, settings)
        linear, _ = tune_linear_ridge(
            frames, sub, split.val, settings, acsf, alphas=tuple(cfg["ridge_alphas"])
        )
        morse = MorsePotential(cutoff=acsf.rc_radial)
        train_potential(morse, frames, sub, split.val, settings)
        for name, model in (("bpnn", bpnn), ("linear", linear), ("morse", morse)):
            batches = prepare_batches(frames, split.test, acsf, settings.batch_size, seed=99)
            m = evaluate_batches(model, batches)
            curves["models"][name].append(
                {
                    "size": size,
                    "e_mae_mev_per_atom": m["e_mae_mev_per_atom"],
                    "f_mae_mev_per_a": m["f_mae_mev_per_a"],
                }
            )
    return curves


def bench_force_weight(frames, cfg) -> dict:
    acsf = _acsf(cfg)
    settings = _settings(cfg)
    split = group_split(
        frames,
        holdout_compositions=tuple(cfg["holdout_compositions"]),
        seed=cfg["split_seed"],
    )
    out = {"weights": cfg["force_weights"], "runs": []}
    for fw in cfg["force_weights"]:
        s = TrainSettings(**{**settings.to_dict(), "force_weight": fw})
        bpnn = make_bpnn(acsf, hidden=tuple(cfg["hidden"]), seed=settings.seed)
        train_potential(bpnn, frames, split.train, split.val, s)
        batches = prepare_batches(frames, split.test, acsf, settings.batch_size, seed=99)
        m = evaluate_batches(bpnn, batches)
        out["runs"].append(
            {
                "force_weight": fw,
                "e_mae_mev_per_atom": m["e_mae_mev_per_atom"],
                "f_mae_mev_per_a": m["f_mae_mev_per_a"],
            }
        )
    return out


def bench_eos(cfg) -> dict:
    from descpot.teacher import get_teacher

    model = load_potential(REPO / "models" / "bpnn_main.pt")
    calc, teacher_info = get_teacher(cfg["teacher"])
    anchors = _load_anchors()
    out = {"teacher": teacher_info, "anchors_source": anchors.get("source"), "compositions": {}}
    for comp in cfg["compositions"]:
        scales = default_scales(comp, n=cfg["points"], span=cfg["span"])
        v_m, e_m = ev_curve_model(model, comp, cfg["reps"], scales, seed=cfg["seed"])
        v_t, e_t = ev_curve_teacher(calc, comp, cfg["reps"], scales, seed=cfg["seed"])
        entry = {
            "volumes_model": v_m.tolist(),
            "energies_model": e_m.tolist(),
            "volumes_teacher": v_t.tolist(),
            "energies_teacher": e_t.tolist(),
            "fit_model": fit_eos(v_m, e_m),
            "fit_teacher": fit_eos(v_t, e_t),
        }
        anchor = anchors.get("entries", {}).get(comp)
        if anchor:
            entry["anchor"] = anchor
        out["compositions"][comp] = entry
    return out


def _load_anchors() -> dict:
    mp = REPO / "data" / "anchors_mp.json"
    lit = REPO / "data" / "anchors_literature.json"
    if mp.exists():
        return json.loads(mp.read_text())
    if lit.exists():
        return json.loads(lit.read_text())
    return {}


def bench_nve(cfg) -> dict:
    model = load_potential(REPO / "models" / "bpnn_main.pt")
    out = {"runs": []}
    for run in cfg["runs"]:
        frame = bcc_supercell(run["composition"], run["reps"], seed=run["seed"])
        t0 = time.perf_counter()
        res = run_nve(
            model,
            frame,
            temperature_k=run["temperature_k"],
            timestep_fs=run["timestep_fs"],
            n_steps=run["steps"],
            seed=run["seed"],
        )
        out["runs"].append(
            {
                **run,
                "n_atoms": frame.n_atoms,
                "drift_mev_per_atom_per_ps": res.drift_mev_per_atom_per_ps,
                "e_tot_std_mev_per_atom": float(np.std(res.e_tot) * 1000),
                "temperature_mean_k": res.temperature_mean_k,
                "wall_seconds": round(time.perf_counter() - t0, 1),
                "times_ps": res.times_ps.tolist(),
                "e_tot": res.e_tot.tolist(),
                "e_pot": res.e_pot.tolist(),
                "e_kin": res.e_kin.tolist(),
            }
        )
    return out


BENCHES = ["main", "split_gap", "learning_curve", "force_weight", "eos", "nve"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=",".join(BENCHES))
    parser.add_argument("--data", default=str(REPO / "data" / "full" / "dataset.extxyz"))
    args = parser.parse_args()
    todo = [b for b in BENCHES if b in args.only.split(",")]
    RESULTS.mkdir(exist_ok=True)
    (REPO / "models").mkdir(exist_ok=True)

    torch.set_num_threads(max(1, torch.get_num_threads()))
    frames = None
    if any(b in todo for b in ("main", "split_gap", "learning_curve", "force_weight")):
        frames = load_frames(args.data)
        print(f"loaded {len(frames)} labeled frames")

    for name in todo:
        cfg = _cfg(name)
        t0 = time.perf_counter()
        if name in ("main", "split_gap", "learning_curve", "force_weight"):
            result = globals()[f"bench_{name}"](frames, cfg)
        else:
            result = globals()[f"bench_{name}"](cfg)
        result["wall_seconds"] = round(time.perf_counter() - t0, 1)
        result["config"] = cfg
        (RESULTS / f"{name}.json").write_text(json.dumps(result, indent=2))
        print(f"[{name}] done in {result['wall_seconds']} s -> results/{name}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
