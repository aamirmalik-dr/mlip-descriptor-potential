"""Operating-point fairness check for the force-error headline.

The main benchmark trains all three models at force weight 0.1, but that
weight was selected by a sweep run on the BPNN. If a baseline's force error
improved a lot at its own force-optimal weight, the headline force gap would
be partly an artifact of the shared operating point. This check retrains the
ridge and Morse baselines at force weight 1.0 (the force-optimal end of the
committed sweep) with the identical data, split, and full epoch budget, and
writes the result next to the main benchmark numbers.

Usage:
    python scripts/check_operating_point.py [--data data/full/dataset.extxyz]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from descpot.data import group_split, load_frames
from descpot.descriptors import AcsfParams
from descpot.model import LinearPotential, MorsePotential
from descpot.training import (
    TrainSettings,
    evaluate_batches,
    prepare_batches,
    train_potential,
)

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(REPO / "data" / "full" / "dataset.extxyz"))
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO / "configs" / "main.yaml").read_text())

    frames = load_frames(args.data)
    split = group_split(
        frames,
        holdout_compositions=tuple(cfg["holdout_compositions"]),
        seed=cfg["split_seed"],
    )
    acsf = AcsfParams()
    settings = TrainSettings(**{**cfg["train"], "force_weight": 1.0})
    # alpha 1e-6 is the winner of the committed ridge sweep, where the val
    # scores for 0, 1e-8, and 1e-6 agree to three decimals
    ridge_alpha = 1e-6

    out = {
        "note": (
            "ridge and Morse retrained at force weight 1.0 (their force-optimal "
            "end of the committed sweep) with the identical data, split, and "
            "full budget; compare against results/main.json at weight 0.1"
        ),
        "force_weight": 1.0,
        "models": {},
    }
    ridge = LinearPotential(acsf)
    s = TrainSettings(**{**settings.to_dict(), "weight_decay": ridge_alpha})
    train_potential(ridge, frames, split.train, split.val, s)
    morse = MorsePotential(cutoff=acsf.rc_radial)
    train_potential(morse, frames, split.train, split.val, settings)
    for name, model in (("linear", ridge), ("morse", morse)):
        m = evaluate_batches(
            model, prepare_batches(frames, split.test, acsf, settings.batch_size, seed=99)
        )
        out["models"][name] = {
            "test_e_mae_mev_per_atom": round(m["e_mae_mev_per_atom"], 2),
            "test_f_mae_mev_per_a": round(m["f_mae_mev_per_a"], 1),
        }
        print(f"{name} at force weight 1.0: {out['models'][name]}")

    (REPO / "results" / "operating_point.json").write_text(json.dumps(out, indent=2))
    print("wrote results/operating_point.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
