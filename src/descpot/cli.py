"""descpot command line interface.

Subcommands:
    predict   Energies and forces for structures in an extxyz file.
    eos       Equation-of-state scan with a trained model.
    md        NVE stability check with a trained model.
    info      Inspect a saved checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def _cmd_predict(args: argparse.Namespace) -> int:
    import torch

    from descpot.data import load_frames
    from descpot.descriptors import build_graph
    from descpot.model import load_potential

    model = load_potential(args.model)
    frames = load_frames(args.xyz)
    out = []
    acsf = model.acsf if hasattr(model, "acsf") else _acsf()
    for k, fr in enumerate(frames):
        graph = build_graph([fr], acsf)
        if args.forces:
            energies, forces = model.energy_forces(graph)
            energies = energies.detach()
            out.append(
                {
                    "frame": k,
                    "energy_ev": float(energies[0]),
                    "energy_ev_per_atom": float(energies[0]) / fr.n_atoms,
                    "forces_ev_per_a": forces.detach().numpy().tolist(),
                }
            )
        else:
            with torch.no_grad():
                energies = model(graph.positions, graph)
            out.append(
                {
                    "frame": k,
                    "energy_ev": float(energies[0]),
                    "energy_ev_per_atom": float(energies[0]) / fr.n_atoms,
                }
            )
    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


def _acsf():
    from descpot.descriptors import AcsfParams

    return AcsfParams()


def _cmd_eos(args: argparse.Namespace) -> int:
    from descpot.eos import default_scales, ev_curve_model, fit_eos
    from descpot.model import load_potential

    model = load_potential(args.model)
    scales = default_scales(args.composition, n=args.points, span=args.span)
    vols, energies = ev_curve_model(model, args.composition, args.reps, scales, seed=args.seed)
    fit = fit_eos(vols, energies)
    fit["composition"] = args.composition
    json.dump(fit, sys.stdout, indent=2)
    print()
    return 0


def _cmd_md(args: argparse.Namespace) -> int:
    from descpot.md import run_nve
    from descpot.model import load_potential
    from descpot.structures import bcc_supercell

    model = load_potential(args.model)
    frame = bcc_supercell(args.composition, args.reps, seed=args.seed)
    result = run_nve(
        model,
        frame,
        temperature_k=args.temperature,
        timestep_fs=args.timestep,
        n_steps=args.steps,
        seed=args.seed,
    )
    json.dump(
        {
            "composition": args.composition,
            "n_atoms": frame.n_atoms,
            "steps": args.steps,
            "timestep_fs": args.timestep,
            "drift_mev_per_atom_per_ps": result.drift_mev_per_atom_per_ps,
            "temperature_mean_k": result.temperature_mean_k,
            "e_tot_std_mev_per_atom": float(np.std(result.e_tot) * 1000),
        },
        sys.stdout,
        indent=2,
    )
    print()
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    from descpot.model import load_potential

    model = load_potential(args.model)
    info = {
        "class": type(model).__name__,
        "n_parameters": model.n_parameters,
        "reference_energies_ev": model.mu.numpy().tolist(),
    }
    if hasattr(model, "acsf"):
        info["acsf"] = model.acsf.to_dict()
        info["n_features"] = model.acsf.n_features
    json.dump(info, sys.stdout, indent=2)
    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="descpot",
        description="From-scratch Behler-Parrinello neural network potential for BCC TiZrNb.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="predict energies (and forces) for an extxyz file")
    p.add_argument("--model", required=True)
    p.add_argument("--xyz", required=True)
    p.add_argument("--forces", action="store_true")
    p.set_defaults(func=_cmd_predict)

    p = sub.add_parser("eos", help="equation-of-state scan and Birch-Murnaghan fit")
    p.add_argument("--model", required=True)
    p.add_argument("--composition", default="TiZrNb")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--points", type=int, default=13)
    p.add_argument("--span", type=float, default=0.06)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_cmd_eos)

    p = sub.add_parser("md", help="NVE molecular dynamics stability check")
    p.add_argument("--model", required=True)
    p.add_argument("--composition", default="TiZrNb")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--temperature", type=float, default=600.0)
    p.add_argument("--timestep", type=float, default=2.0)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_cmd_md)

    p = sub.add_parser("info", help="inspect a saved checkpoint")
    p.add_argument("--model", required=True)
    p.set_defaults(func=_cmd_info)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
