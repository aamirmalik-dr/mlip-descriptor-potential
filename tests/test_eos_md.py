"""EOS fitting and NVE integrator checks on a cheap analytic potential."""

import numpy as np
import pytest
import torch

from descpot.eos import birch_murnaghan, ev_curve_model, fit_eos
from descpot.md import run_nve
from descpot.model import MorsePotential
from descpot.structures import bcc_supercell


def _morse_model():
    torch.manual_seed(0)
    model = MorsePotential()
    with torch.no_grad():
        model.raw_depth.fill_(-0.5)
        model.raw_width.fill_(0.6)
        model.raw_r0.fill_(0.9)
    model.set_reference(np.array([-4.0, -4.5, -5.0]))
    model.eval()
    return model


def test_birch_murnaghan_fit_recovers_parameters():
    v = np.linspace(14.0, 20.0, 15)
    e = birch_murnaghan(v, e0=-7.5, v0=17.0, b0=0.8, b0p=4.2)
    fit = fit_eos(v, e)
    assert fit["e0_ev_per_atom"] == pytest.approx(-7.5, abs=1e-8)
    assert fit["v0_a3_per_atom"] == pytest.approx(17.0, abs=1e-6)
    assert fit["b0_gpa"] == pytest.approx(0.8 * 160.2176634, rel=1e-6)
    assert fit["a0_bcc_a"] == pytest.approx((2 * 17.0) ** (1 / 3), abs=1e-8)


def test_ev_curve_has_interior_minimum():
    model = _morse_model()
    scales = np.linspace(0.92, 1.10, 10)
    vols, energies = ev_curve_model(model, "Nb", 2, scales, seed=0)
    k = int(np.argmin(energies))
    assert 0 < k < len(scales) - 1
    assert np.all(np.diff(vols) > 0)


def test_nve_energy_conservation_morse():
    model = _morse_model()
    frame = bcc_supercell("Nb", 2, seed=0)
    result = run_nve(model, frame, temperature_k=300.0, timestep_fs=2.0, n_steps=200, seed=0)
    # a symplectic integrator with consistent forces holds total energy tightly
    assert abs(result.drift_mev_per_atom_per_ps) < 2.0
    assert float(np.std(result.e_tot)) * 1000 < 1.0
    assert result.temperature_mean_k > 50.0


def test_nve_smaller_timestep_smaller_drift():
    model = _morse_model()
    frame = bcc_supercell("Nb", 2, seed=1)
    drifts = {}
    for dt in (4.0, 1.0):
        res = run_nve(model, frame, temperature_k=300.0, timestep_fs=dt, n_steps=150, seed=0)
        drifts[dt] = float(np.std(res.e_tot))
    assert drifts[1.0] <= drifts[4.0] + 1e-9
