"""CLI smoke tests on a tiny saved checkpoint."""

import json

import numpy as np
import torch

from descpot.cli import main
from descpot.data import save_frames
from descpot.model import MorsePotential, save_potential
from descpot.structures import bcc_supercell


def _checkpoint(tmp_path):
    torch.manual_seed(0)
    model = MorsePotential()
    with torch.no_grad():
        model.raw_depth.fill_(-0.5)
        model.raw_width.fill_(0.6)
        model.raw_r0.fill_(0.9)
    model.set_reference(np.array([-4.0, -4.5, -5.0]))
    path = tmp_path / "morse.pt"
    save_potential(model, path)
    return path


def test_cli_info(tmp_path, capsys):
    path = _checkpoint(tmp_path)
    assert main(["info", "--model", str(path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["class"] == "MorsePotential"
    assert out["n_parameters"] == 21


def test_cli_predict_with_forces(tmp_path, capsys):
    path = _checkpoint(tmp_path)
    frames = [bcc_supercell("TiZrNb", 2, seed=0)]
    xyz = tmp_path / "frames.extxyz"
    save_frames(frames, xyz)
    assert main(["predict", "--model", str(path), "--xyz", str(xyz), "--forces"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out) == 1
    assert out[0]["energy_ev"] < 0
    assert len(out[0]["forces_ev_per_a"]) == 16


def test_cli_eos_and_md(tmp_path, capsys):
    path = _checkpoint(tmp_path)
    assert main(["eos", "--model", str(path), "--composition", "Nb", "--points", "9"]) == 0
    eos = json.loads(capsys.readouterr().out)
    assert eos["b0_gpa"] > 0
    assert 2.5 < eos["a0_bcc_a"] < 4.5
    assert main(["md", "--model", str(path), "--composition", "Nb", "--steps", "50"]) == 0
    md = json.loads(capsys.readouterr().out)
    assert "drift_mev_per_atom_per_ps" in md
    assert md["n_atoms"] == 16
