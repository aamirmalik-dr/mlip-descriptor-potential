"""Generate and teacher-label the BCC TiZrNb training set.

Builds rattle, strain, and teacher-driven MD batches for every composition in
the config, labels all frames with the surrogate teacher (a pretrained
universal potential, NOT DFT), and writes:

    data/full/dataset.extxyz   the full labeled dataset (gitignored)
    data/provenance.json       teacher identity, counts, and timings

Usage:
    python scripts/generate_data.py --config configs/dataset.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from descpot.data import save_frames
from descpot.structures import rattle_batch, strain_batch
from descpot.teacher import get_teacher, label_frames, teacher_md_batch

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO / "configs" / "dataset.yaml"))
    parser.add_argument("--out", default=str(REPO / "data" / "full" / "dataset.extxyz"))
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    calc, teacher_info = get_teacher(cfg["teacher"])
    frames = []
    md_seconds = 0.0
    for ci, comp in enumerate(cfg["compositions"] + cfg["holdout_compositions"]):
        seed_base = cfg["seed"] + 1000 * ci
        for b, amp in enumerate(cfg["rattle_amplitudes"]):
            frames.extend(
                rattle_batch(
                    comp,
                    cfg["reps_small"],
                    amp,
                    cfg["rattle_frames_per_batch"],
                    seed=seed_base + b,
                    batch=b,
                )
            )
        # one rattle batch on the larger cell for size diversity
        frames.extend(
            rattle_batch(
                comp,
                cfg["reps_large"],
                cfg["rattle_amplitudes"][1],
                cfg["rattle_frames_large"],
                seed=seed_base + 50,
                batch=9,
            )
        )
        frames.extend(
            strain_batch(
                comp,
                cfg["reps_small"],
                cfg["strain_frames_per_batch"],
                seed=seed_base + 100,
                batch=0,
            )
        )
        t0 = time.perf_counter()
        for b, temp in enumerate(cfg["md_temperatures_k"]):
            frames.extend(
                teacher_md_batch(
                    comp,
                    cfg["reps_small"],
                    calc,
                    temperature_k=temp,
                    n_steps=cfg["md_steps"],
                    sample_every=cfg["md_sample_every"],
                    seed=seed_base + 200 + b,
                    batch=b,
                )
            )
        md_seconds += time.perf_counter() - t0

    sec_per_frame = label_frames(frames, calc)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_frames(frames, out)

    groups = sorted({fr.group for fr in frames})
    provenance = {
        "label_source": "surrogate model labels from a pretrained universal potential, NOT DFT",
        "teacher": teacher_info,
        "n_frames": len(frames),
        "n_groups": len(groups),
        "compositions": sorted({fr.composition for fr in frames}),
        "atoms_min": min(fr.n_atoms for fr in frames),
        "atoms_max": max(fr.n_atoms for fr in frames),
        "labeling_seconds_per_frame": round(sec_per_frame, 4),
        "md_generation_seconds": round(md_seconds, 1),
        "config": cfg,
    }
    (REPO / "data" / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps({k: v for k, v in provenance.items() if k != "config"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
