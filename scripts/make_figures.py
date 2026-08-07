"""Render all committed figures from results/*.json.

Colors follow one fixed assignment everywhere: BPNN blue, linear ridge orange,
Morse aqua, teacher dark gray. Anchors appear as reference markers.

Usage:
    python scripts/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
FIGS = RESULTS / "figures"

C_BPNN = "#2a78d6"
C_LINEAR = "#eb6834"
C_MORSE = "#1baf7a"
C_TEACHER = "#52514e"
C_ANCHOR = "#4a3aa7"
C_GRID = "#e1e0d9"
C_MUTED = "#898781"
LABELS = {"bpnn": "BPNN", "linear": "ridge (linear)", "morse": "Morse (pair)"}
COLORS = {"bpnn": C_BPNN, "linear": C_LINEAR, "morse": C_MORSE}

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 9,
        "axes.edgecolor": C_MUTED,
        "axes.labelcolor": "#0b0b0b",
        "axes.grid": True,
        "grid.color": C_GRID,
        "grid.linewidth": 0.6,
        "xtick.color": C_MUTED,
        "ytick.color": C_MUTED,
        "axes.titlesize": 9.5,
        "legend.frameon": False,
        "savefig.facecolor": "white",
    }
)


def _load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def _parity_panel(ax, ev, color, label, kind: str) -> None:
    if kind == "energy":
        x = np.array(ev["e_true_per_atom"])
        y = np.array(ev["e_pred_per_atom"])
        mae = ev["e_mae_mev_per_atom"]
        unit = f"MAE {mae:.1f} meV/atom"
    else:
        x = np.array(ev["f_true_sample"])
        y = np.array(ev["f_pred_sample"])
        mae = ev["f_mae_mev_per_a"]
        unit = f"MAE {mae:.0f} meV/A"
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    pad = 0.05 * (hi - lo)
    lims = (lo - pad, hi + pad)
    ax.plot(lims, lims, ls="--", lw=0.9, color=C_MUTED, zorder=1)
    ax.scatter(x, y, s=7, alpha=0.55, color=color, edgecolors="none", zorder=2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title(f"{label}\n{unit}")
    ax.set_aspect("equal")


def fig_hero() -> None:
    main = _load("main")
    lc = _load("learning_curve")
    ev = main["models"]["bpnn"]["eval"]["test"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))

    _parity_panel(axes[0], ev, C_BPNN, "BPNN energy parity (held-out groups)", "energy")
    axes[0].set_xlabel("teacher energy (eV/atom)")
    axes[0].set_ylabel("BPNN energy (eV/atom)")

    _parity_panel(axes[1], ev, C_BPNN, "BPNN force parity (held-out groups)", "force")
    axes[1].set_xlabel("teacher force $F_x$ (eV/A)")
    axes[1].set_ylabel("BPNN force $F_x$ (eV/A)")

    ax = axes[2]
    for name in ("bpnn", "linear", "morse"):
        pts = lc["models"][name]
        sizes = [p["size"] for p in pts]
        maes = [p["e_mae_mev_per_atom"] for p in pts]
        ax.plot(sizes, maes, "o-", lw=1.8, ms=4.5, color=COLORS[name], label=LABELS[name])
        ax.annotate(
            f"{maes[-1]:.1f}",
            (sizes[-1], maes[-1]),
            textcoords="offset points",
            xytext=(6, -2),
            fontsize=8,
            color=COLORS[name],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training frames")
    ax.set_ylabel("test energy MAE (meV/atom)")
    ax.set_title("learning curve (matched budget)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "hero.png", bbox_inches="tight")
    plt.close(fig)


def fig_parity_grid() -> None:
    main = _load("main")
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.0))
    for col, name in enumerate(("bpnn", "linear", "morse")):
        ev = main["models"][name]["eval"]["test"]
        _parity_panel(axes[0, col], ev, COLORS[name], LABELS[name], "energy")
        _parity_panel(axes[1, col], ev, COLORS[name], LABELS[name], "force")
        axes[0, col].set_xlabel("teacher (eV/atom)")
        axes[1, col].set_xlabel("teacher $F_x$ (eV/A)")
    axes[0, 0].set_ylabel("predicted (eV/atom)")
    axes[1, 0].set_ylabel("predicted $F_x$ (eV/A)")
    fig.suptitle("energy (top) and force (bottom) parity on held-out generation groups", y=0.99)
    fig.tight_layout()
    fig.savefig(FIGS / "parity_grid.png", bbox_inches="tight")
    plt.close(fig)


def fig_learning_curve() -> None:
    lc = _load("learning_curve")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for ax, key, ylabel in (
        (axes[0], "e_mae_mev_per_atom", "test energy MAE (meV/atom)"),
        (axes[1], "f_mae_mev_per_a", "test force MAE (meV/A)"),
    ):
        for name in ("bpnn", "linear", "morse"):
            pts = lc["models"][name]
            ax.plot(
                [p["size"] for p in pts],
                [p[key] for p in pts],
                "o-",
                lw=1.8,
                ms=4.5,
                color=COLORS[name],
                label=LABELS[name],
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("training frames")
        ax.set_ylabel(ylabel)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "learning_curve.png", bbox_inches="tight")
    plt.close(fig)


def fig_force_weight() -> None:
    fw = _load("force_weight")
    runs = fw["runs"]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    xs = [r["f_mae_mev_per_a"] for r in runs]
    ys = [r["e_mae_mev_per_atom"] for r in runs]
    ax.scatter(xs, ys, s=42, color=C_BPNN, zorder=2)
    for r in runs:
        ax.annotate(
            f"$w_F$={r['force_weight']:g}",
            (r["f_mae_mev_per_a"], r["e_mae_mev_per_atom"]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=8.5,
            color="#0b0b0b",
        )
    ax.set_xlabel("test force MAE (meV/A)")
    ax.set_ylabel("test energy MAE (meV/atom)")
    ax.set_title("force-weight tradeoff (BPNN, same data and budget)")
    fig.tight_layout()
    fig.savefig(FIGS / "force_weight.png", bbox_inches="tight")
    plt.close(fig)


def fig_split_gap() -> None:
    main = _load("main")
    gap = _load("split_gap")
    transfer = main["models"]["bpnn"]["eval"]["transfer"]
    cases = [
        ("group split\n(3 split seeds)", gap["group_mean"], C_BPNN),
        ("random-frame split\n(optimistic control)", gap["random_frame_mean"], C_MUTED),
        ("held-out composition\n(transfer)", None, C_ANCHOR),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    x = np.arange(len(cases))
    for ax, key, tkey, ylabel in (
        (axes[0], "e_mae_mev_per_atom", "e_mae_mev_per_atom", "energy MAE (meV/atom)"),
        (axes[1], "f_mae_mev_per_a", "f_mae_mev_per_a", "force MAE (meV/A)"),
    ):
        std_key = "e_mae_std" if "atom" in key else "f_mae_std"
        vals = [m[key] if m is not None else transfer[tkey] for _, m, _c in cases]
        errs = [m[std_key] if m is not None else 0.0 for _, m, _c in cases]
        bars = ax.bar(
            x,
            vals,
            width=0.55,
            color=[c for _, _, c in cases],
            yerr=errs,
            capsize=4,
            error_kw={"ecolor": "#0b0b0b", "elinewidth": 1.0},
        )
        for b, v, e in zip(bars, vals, errs):
            ax.annotate(
                f"{v:.1f}",
                (b.get_x() + b.get_width() / 2, v + e),
                ha="center",
                va="bottom",
                fontsize=8.5,
                xytext=(0, 3),
                textcoords="offset points",
            )
        ax.set_xticks(x, [c[0] for c in cases], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
    fig.suptitle("what the split protocol claims: same BPNN, same budget", y=1.0)
    fig.tight_layout()
    fig.savefig(FIGS / "split_gap.png", bbox_inches="tight")
    plt.close(fig)


def fig_eos() -> None:
    eos = _load("eos")
    comps = list(eos["compositions"].keys())
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0))
    for ax, comp in zip(axes.ravel(), comps):
        entry = eos["compositions"][comp]
        vm = np.array(entry["volumes_model"])
        em = np.array(entry["energies_model"])
        vt = np.array(entry["volumes_teacher"])
        et = np.array(entry["energies_teacher"])
        ax.plot(vt, et - et.min(), "s--", ms=3.5, lw=1.4, color=C_TEACHER, label="teacher (CHGNet)")
        ax.plot(vm, em - et.min(), "o-", ms=3.5, lw=1.6, color=C_BPNN, label="BPNN (this work)")
        fm, ft = entry["fit_model"], entry["fit_teacher"]
        txt = (
            f"a0: {fm['a0_bcc_a']:.3f} vs {ft['a0_bcc_a']:.3f} A\n"
            f"B0: {fm['b0_gpa']:.0f} vs {ft['b0_gpa']:.0f} GPa"
        )
        if "anchor" in entry:
            a = entry["anchor"]
            v_anchor = a["a0_bcc_a"] ** 3 / 2.0
            ax.axvline(v_anchor, color=C_ANCHOR, lw=1.2, ls=":", zorder=1)
            ax.annotate(
                f"DFT anchor a0 {a['a0_bcc_a']:.2f}",
                (v_anchor, ax.get_ylim()[1] * 0.05),
                rotation=90,
                fontsize=7.5,
                color=C_ANCHOR,
                ha="right",
                va="bottom",
            )
            txt += f"\nanchor: {a['a0_bcc_a']:.2f} A, {a['b0_gpa']:.0f} GPa"
        ax.set_title(f"BCC {comp}")
        ax.text(
            0.03,
            0.97,
            txt,
            transform=ax.transAxes,
            fontsize=7.5,
            va="top",
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": C_GRID},
        )
        ax.set_xlabel("volume (A$^3$/atom)")
        ax.set_ylabel("relative energy (eV/atom)")
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle("equation of state: student vs teacher vs DFT anchors", y=0.995)
    fig.tight_layout()
    fig.savefig(FIGS / "eos.png", bbox_inches="tight")
    plt.close(fig)


def fig_nve() -> None:
    nve = _load("nve")
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    shades = [C_BPNN, C_LINEAR, C_MORSE]
    for run, color in zip(nve["runs"], shades):
        t = np.array(run["times_ps"])
        e = (np.array(run["e_tot"]) - run["e_tot"][0]) * 1000.0
        label = (
            f"{run['composition']} {run['temperature_k']:.0f} K, "
            f"drift {run['drift_mev_per_atom_per_ps']:+.3f} meV/atom/ps"
        )
        ax.plot(t, e, lw=1.4, color=color, label=label)
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("$E_{tot}$ change (meV/atom)")
    ax.set_title("NVE total-energy conservation, BPNN forces, 2 fs timestep")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "nve.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_hero()
    fig_parity_grid()
    fig_learning_curve()
    fig_force_weight()
    fig_split_gap()
    fig_eos()
    fig_nve()
    print(f"figures written to {FIGS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
