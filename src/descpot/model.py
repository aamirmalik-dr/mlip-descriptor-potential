"""Potential models: BPNN, linear (ridge-style), and pairwise Morse baseline.

All three share one interface: forward(positions, graph) returns per-structure
total energies in eV, and energy_forces(graph) adds forces from autograd.
Energies are modeled as a fixed composition-linear reference (per-element
reference energies, fit on the training set) plus a learned residual.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from descpot.descriptors import (
    _PAIR_INDEX,
    N_ELEMENTS,
    AcsfParams,
    Graph,
    _cutoff_fn,
    compute_descriptors,
)


class PotentialBase(nn.Module):
    """Shared reference-energy handling and autograd force evaluation."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mu", torch.zeros(N_ELEMENTS, dtype=torch.float64))

    def set_reference(self, mu: np.ndarray) -> None:
        """Set fixed per-element reference energies (eV/atom)."""
        self.mu.copy_(torch.tensor(mu, dtype=torch.float64))

    def residual_atomic_energies(self, positions: torch.Tensor, graph: Graph) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, positions: torch.Tensor, graph: Graph) -> torch.Tensor:
        """Per-structure total energies in eV, shape (S,)."""
        e_atomic = self.residual_atomic_energies(positions, graph)
        e_atomic = e_atomic + self.mu[graph.species]
        out = torch.zeros(graph.n_structs, dtype=positions.dtype)
        return out.index_add(0, graph.struct_of_atom, e_atomic)

    def energy_forces(self, graph: Graph) -> tuple[torch.Tensor, torch.Tensor]:
        """Energies (S,) and forces (N, 3) with forces from autograd."""
        pos = graph.positions.detach().clone().requires_grad_(True)
        energies = self.forward(pos, graph)
        (grad,) = torch.autograd.grad(energies.sum(), pos, create_graph=self.training)
        return energies, -grad

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DescriptorScaler(nn.Module):
    """Per-element feature standardization, fit once on training atoms."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(N_ELEMENTS, n_features, dtype=torch.float64))
        self.register_buffer("std", torch.ones(N_ELEMENTS, n_features, dtype=torch.float64))

    def fit(self, descriptors: torch.Tensor, species: torch.Tensor) -> None:
        for t in range(N_ELEMENTS):
            sel = descriptors[species == t]
            if len(sel) > 0:
                self.mean[t] = sel.mean(dim=0)
                std = sel.std(dim=0)
                # a feature that is constant in training (for example a
                # neighbor-element channel absent from the training
                # compositions) gets unit scale, not a tiny clamp: dividing by
                # a near-zero std would explode the first time the feature is
                # nonzero at inference
                self.std[t] = torch.where(std < 1e-6, torch.ones_like(std), std)

    def forward(self, descriptors: torch.Tensor, species: torch.Tensor) -> torch.Tensor:
        return (descriptors - self.mean[species]) / self.std[species]


class BpnnPotential(PotentialBase):
    """Behler-Parrinello network: ACSF descriptors into per-element MLPs.

    Args:
        acsf: Symmetry-function parameters.
        hidden: Hidden-layer widths of each element network.
    """

    def __init__(self, acsf: AcsfParams, hidden: tuple[int, ...] = (32, 32)) -> None:
        super().__init__()
        self.acsf = acsf
        self.hidden = tuple(hidden)
        self.scaler = DescriptorScaler(acsf.n_features)
        self.nets = nn.ModuleList()
        for _ in range(N_ELEMENTS):
            layers: list[nn.Module] = []
            width_in = acsf.n_features
            for width in hidden:
                layers.append(nn.Linear(width_in, width, dtype=torch.float64))
                layers.append(nn.SiLU())
                width_in = width
            layers.append(nn.Linear(width_in, 1, dtype=torch.float64))
            self.nets.append(nn.Sequential(*layers))
        for net in self.nets:
            last = net[-1]
            nn.init.zeros_(last.bias)
            nn.init.normal_(last.weight, std=1e-2)

    def residual_atomic_energies(self, positions: torch.Tensor, graph: Graph) -> torch.Tensor:
        desc = compute_descriptors(positions, graph, self.acsf)
        x = self.scaler(desc, graph.species)
        out = torch.zeros(len(graph.species), dtype=positions.dtype)
        for t in range(N_ELEMENTS):
            mask = graph.species == t
            if mask.any():
                out = out.index_add(0, mask.nonzero().squeeze(1), self.nets[t](x[mask]).squeeze(1))
        return out


class LinearPotential(PotentialBase):
    """Ridge-style linear model on the identical ACSF descriptors.

    Trained with the same loss and budget as the BPNN (weight decay plays the
    ridge alpha role), so the comparison isolates what the nonlinearity buys.
    """

    def __init__(self, acsf: AcsfParams) -> None:
        super().__init__()
        self.acsf = acsf
        self.scaler = DescriptorScaler(acsf.n_features)
        self.weight = nn.Parameter(torch.zeros(N_ELEMENTS, acsf.n_features, dtype=torch.float64))
        self.bias = nn.Parameter(torch.zeros(N_ELEMENTS, dtype=torch.float64))

    def residual_atomic_energies(self, positions: torch.Tensor, graph: Graph) -> torch.Tensor:
        desc = compute_descriptors(positions, graph, self.acsf)
        x = self.scaler(desc, graph.species)
        return (x * self.weight[graph.species]).sum(dim=1) + self.bias[graph.species]


class MorsePotential(PotentialBase):
    """Pairwise Morse potential with per-element-pair parameters.

    The many-body-blind baseline: six (D, a, r0) triples, one per unordered
    element pair, plus per-element offsets, evaluated on the same neighbor
    lists with a smooth cosine cutoff.
    """

    def __init__(self, cutoff: float = 5.5) -> None:
        super().__init__()
        self.cutoff = float(cutoff)
        self.raw_depth = nn.Parameter(torch.full((6,), -1.0, dtype=torch.float64))
        self.raw_width = nn.Parameter(torch.full((6,), 0.5, dtype=torch.float64))
        self.raw_r0 = nn.Parameter(torch.full((6,), 1.0, dtype=torch.float64))
        self.offset = nn.Parameter(torch.zeros(N_ELEMENTS, dtype=torch.float64))
        self.register_buffer(
            "pair_table", torch.tensor(_PAIR_INDEX, dtype=torch.long), persistent=False
        )

    def residual_atomic_energies(self, positions: torch.Tensor, graph: Graph) -> torch.Tensor:
        vec = positions[graph.pair_j] + graph.shift_vec - positions[graph.pair_i]
        r = torch.linalg.norm(vec, dim=1)
        ptype = self.pair_table[graph.species[graph.pair_i], graph.species[graph.pair_j]]
        depth = nn.functional.softplus(self.raw_depth)[ptype]
        width = nn.functional.softplus(self.raw_width)[ptype]
        r0 = 2.0 + nn.functional.softplus(self.raw_r0)[ptype]
        x = torch.exp(-width * (r - r0))
        pair_e = depth * (x * x - 2.0 * x) * _cutoff_fn(r, self.cutoff)
        out = torch.zeros(len(graph.species), dtype=positions.dtype)
        out = out.index_add(0, graph.pair_i, 0.5 * pair_e)
        return out + self.offset[graph.species]


def save_potential(model: PotentialBase, path: str | Path, extra: dict | None = None) -> None:
    """Serialize a potential with enough metadata to reconstruct it."""
    meta: dict = {"extra": extra or {}}
    if isinstance(model, BpnnPotential):
        meta.update({"kind": "bpnn", "acsf": model.acsf.to_dict(), "hidden": list(model.hidden)})
    elif isinstance(model, LinearPotential):
        meta.update({"kind": "linear", "acsf": model.acsf.to_dict()})
    elif isinstance(model, MorsePotential):
        meta.update({"kind": "morse", "cutoff": model.cutoff})
    else:
        raise TypeError(f"unknown potential type {type(model)}")
    torch.save({"meta_json": json.dumps(meta), "state": model.state_dict()}, str(path))


def load_potential(path: str | Path) -> PotentialBase:
    """Load a potential saved by save_potential."""
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    meta = json.loads(payload["meta_json"])
    if meta["kind"] == "bpnn":
        model = BpnnPotential(AcsfParams.from_dict(meta["acsf"]), tuple(meta["hidden"]))
    elif meta["kind"] == "linear":
        model = LinearPotential(AcsfParams.from_dict(meta["acsf"]))
    elif meta["kind"] == "morse":
        model = MorsePotential(cutoff=meta["cutoff"])
    else:
        raise ValueError(f"unknown potential kind {meta['kind']}")
    model.load_state_dict(payload["state"])
    model.eval()
    return model
