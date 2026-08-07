"""Carve the committed sample from the full labeled dataset.

Takes a stratified sample (a few frames from every generation group) so the
committed extxyz demonstrates the whole variety while staying small.

Usage:
    python scripts/make_sample_data.py [--per-group 3]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from descpot.data import load_frames, save_frames

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(REPO / "data" / "full" / "dataset.extxyz"))
    parser.add_argument("--out", default=str(REPO / "data" / "sample_frames.extxyz"))
    parser.add_argument("--per-group", type=int, default=3)
    args = parser.parse_args()

    frames = load_frames(args.data)
    by_group = defaultdict(list)
    for fr in frames:
        by_group[fr.group].append(fr)
    sample = []
    for group in sorted(by_group):
        sample.extend(by_group[group][: args.per_group])
    save_frames(sample, args.out)
    size_kb = Path(args.out).stat().st_size / 1024
    print(f"{len(sample)} frames from {len(by_group)} groups -> {args.out} ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
