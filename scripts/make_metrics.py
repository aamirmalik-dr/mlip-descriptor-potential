"""Aggregate benchmark results into results/metrics.json (the headline file).

Usage:
    python scripts/make_metrics.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"


def _load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def main() -> int:
    main_r = _load("main")
    gap = _load("split_gap")
    lc = _load("learning_curve")
    fw = _load("force_weight")
    eos = _load("eos")
    nve = _load("nve")
    provenance = json.loads((REPO / "data" / "provenance.json").read_text())

    models = {}
    for name, entry in main_r["models"].items():
        test = entry["eval"]["test"]
        transfer = entry["eval"].get("transfer", {})
        train_seconds = entry.get("train_seconds")
        if train_seconds is None:
            train_seconds = entry.get("history", {}).get("train_seconds", 0.0)
        models[name] = {
            "n_parameters": entry["n_parameters"],
            "train_seconds": round(train_seconds, 1),
            "test_e_mae_mev_per_atom": round(test["e_mae_mev_per_atom"], 2),
            "test_f_mae_mev_per_a": round(test["f_mae_mev_per_a"], 1),
            "test_e_rmse_mev_per_atom": round(test["e_rmse_mev_per_atom"], 2),
            "test_f_rmse_mev_per_a": round(test["f_rmse_mev_per_a"], 1),
            "transfer_e_mae_mev_per_atom": (
                round(transfer["e_mae_mev_per_atom"], 2) if transfer else None
            ),
            "transfer_f_mae_mev_per_a": round(transfer["f_mae_mev_per_a"], 1) if transfer else None,
        }

    eos_summary = {}
    for comp, entry in eos["compositions"].items():
        row = {
            "a0_model": round(entry["fit_model"]["a0_bcc_a"], 3),
            "a0_teacher": round(entry["fit_teacher"]["a0_bcc_a"], 3),
            "b0_model_gpa": round(entry["fit_model"]["b0_gpa"], 1),
            "b0_teacher_gpa": round(entry["fit_teacher"]["b0_gpa"], 1),
        }
        if "anchor" in entry:
            row["a0_anchor_dft"] = entry["anchor"]["a0_bcc_a"]
            row["b0_anchor_dft_gpa"] = entry["anchor"].get("b0_gpa")
            row["anchor_id"] = entry["anchor"].get("material_id")
        eos_summary[comp] = row

    metrics = {
        "label_provenance": {
            "training_labels": "surrogate model labels from "
            + provenance["teacher"]["package"]
            + " "
            + provenance["teacher"]["version"]
            + " ("
            + provenance["teacher"]["checkpoint"]
            + "), NOT DFT",
            "teacher_license": provenance["teacher"]["license"],
            "dft_anchors": eos.get("anchors_source"),
        },
        "dataset": {
            "n_frames": provenance["n_frames"],
            "n_groups": provenance["n_groups"],
            "atoms_min": provenance["atoms_min"],
            "atoms_max": provenance["atoms_max"],
            "labeling_seconds_per_frame": provenance["labeling_seconds_per_frame"],
            "split": main_r["split"],
        },
        "models": models,
        "split_gap_bpnn": {
            "seeds": gap["seeds"],
            "group_e_mae_mean": round(gap["group_mean"]["e_mae_mev_per_atom"], 2),
            "group_e_mae_std": round(gap["group_mean"]["e_mae_std"], 2),
            "random_frame_e_mae_mean": round(gap["random_frame_mean"]["e_mae_mev_per_atom"], 2),
            "random_frame_e_mae_std": round(gap["random_frame_mean"]["e_mae_std"], 2),
            "group_f_mae_mean": round(gap["group_mean"]["f_mae_mev_per_a"], 1),
            "group_f_mae_std": round(gap["group_mean"]["f_mae_std"], 1),
            "random_frame_f_mae_mean": round(gap["random_frame_mean"]["f_mae_mev_per_a"], 1),
            "random_frame_f_mae_std": round(gap["random_frame_mean"]["f_mae_std"], 1),
            "per_seed": {k: gap[k] for k in ("group", "random_frame")},
        },
        "learning_curve_sizes": lc["sizes"],
        "force_weight_sweep": fw["runs"],
        "eos": eos_summary,
        "nve": [
            {
                "composition": r["composition"],
                "temperature_k": r["temperature_k"],
                "steps": r["steps"],
                "timestep_fs": r["timestep_fs"],
                "drift_mev_per_atom_per_ps": round(r["drift_mev_per_atom_per_ps"], 3),
                "e_tot_std_mev_per_atom": round(r["e_tot_std_mev_per_atom"], 3),
            }
            for r in nve["runs"]
        ],
        "wall_seconds": {
            name: _load(name).get("wall_seconds")
            for name in ("main", "split_gap", "learning_curve", "force_weight", "eos", "nve")
        },
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics["models"], indent=2))
    print("split gap:", json.dumps(metrics["split_gap_bpnn"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
