"""Reproduce everything: data, benchmarks, figures, metrics.

Requires the teacher extra (pip install -e ".[teacher]") and roughly an hour
of CPU. Individual stages can be re-run separately; see each script's help.

Usage:
    python scripts/run_all.py [--skip-data]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run(script: str, *args: str) -> None:
    cmd = [sys.executable, str(REPO / "scripts" / script), *args]
    print(f"\n== {' '.join(cmd[1:])} ==")
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-data", action="store_true", help="reuse data/full/dataset.extxyz")
    args = parser.parse_args()
    if not args.skip_data:
        run("generate_data.py")
        run("make_sample_data.py")
    run("fetch_mp_anchors.py")
    run("run_benchmarks.py")
    run("make_figures.py")
    run("make_metrics.py")
    print("\nall stages complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
