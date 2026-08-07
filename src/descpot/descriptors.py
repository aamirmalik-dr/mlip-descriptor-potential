"""Atom-centered symmetry functions (Behler-Parrinello ACSF) in PyTorch.

Implements element-resolved radial G2 and angular G4 symmetry functions from
scratch, differentiable with respect to atomic positions so that forces come
from autograd. Structures are packed into a flat Graph so batches of frames
evaluate in single vectorized tensor ops.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from descpot.neighbors import neighbor_pairs, restrict_pairs, triplets_from_pairs

N_ELEMENTS = 3  # Ti, Zr, Nb

# Unordered element-pair index table for angular functions: (t1, t2) -> 0..5
_PAIR_INDEX = np.array(
    [
        [0, 1, 2],
        [1, 3, 4],
        [2, 4, 5],
    ]
)
N_PAIR_TYPES = 6


def _default_radial() -> list[tuple[float, float]]:
    """Shifted-Gaussian (rs, eta) grid covering the first three BCC shells."""
    centers = np.linspace(2.2, 5.2, 8)
    delta = centers[1] - centers[0]
    eta = 1.0 / (2.0 * delta * delta)
    return [(float(rs), float(eta)) for rs in centers]


def _default_angular() -> list[tuple[float, float, float]]:
    """(eta, zeta, lambda) sets, standard narrow/wide angular resolution."""
    return [
        (0.02, 1.0, 1.0),
        (0.02, 1.0, -1.0),
        (0.02, 4.0, 1.0),
        (0.02, 4.0, -1.0),
    ]


@dataclass
class AcsfParams:
    """Symmetry-function hyperparameters.

    Attributes:
        rc_radial: Radial cutoff in Angstrom.
        rc_angular: Angular cutoff in Angstrom (shorter, triplets are cubic in
            neighbor count).
        radial: List of (rs, eta) shifted-Gaussian parameters.
        angular: List of (eta, zeta, lambda) G4 parameters.
    """

    rc_radial: float = 5.5
    rc_angular: float = 4.5
    radial: list[tuple[float, float]] = field(default_factory=_default_radial)
    angular: list[tuple[float, float, float]] = field(default_factory=_default_angular)

    @property
    def n_features(self) -> int:
        return N_ELEMENTS * len(self.radial) + N_PAIR_TYPES * len(self.angular)

    def to_dict(self) -> dict:
        return {
            "rc_radial": self.rc_radial,
            "rc_angular": self.rc_angular,
            "radial": [list(p) for p in self.radial],
            "angular": [list(p) for p in self.angular],
        }

    @staticmethod
    def from_dict(d: dict) -> AcsfParams:
        return AcsfParams(
            rc_radial=float(d["rc_radial"]),
            rc_angular=float(d["rc_angular"]),
            radial=[tuple(p) for p in d["radial"]],
            angular=[tuple(p) for p in d["angular"]],
        )


@dataclass
class Graph:
    """Flat tensor view of one or more structures for descriptor evaluation.

    Atom indices of every structure are offset into one concatenated array;
    pair shift vectors are premultiplied by each structure's own cell, so
    strained and unstrained frames batch together.
    """

    positions: torch.Tensor  # (N, 3) float
    species: torch.Tensor  # (N,) long
    pair_i: torch.Tensor  # (P,) long
    pair_j: torch.Tensor  # (P,) long
    shift_vec: torch.Tensor  # (P, 3) float, constant
    ang_keep: torch.Tensor  # (Pa,) long indices into pairs within rc_angular
    tri_ij: torch.Tensor  # (T,) long indices into pairs
    tri_ik: torch.Tensor  # (T,) long indices into pairs
    struct_of_atom: torch.Tensor  # (N,) long
    n_structs: int
    n_atoms_per_struct: torch.Tensor  # (S,) long


def build_graph(frames, params: AcsfParams, dtype=torch.float64) -> Graph:
    """Pack a list of Frame objects into one flat Graph.

    Args:
        frames: Iterable of descpot.data.Frame.
        params: Symmetry-function parameters (sets both cutoffs).
        dtype: Torch float dtype for positions and shift vectors.

    Returns:
        Graph ready for compute_descriptors / potential evaluation.
    """
    pos_list, spec_list, pi_list, pj_list, sv_list = [], [], [], [], []
    keep_list, tij_list, tik_list, soa_list, nat_list = [], [], [], [], []
    atom_off = 0
    pair_off = 0
    for s, fr in enumerate(frames):
        pairs = neighbor_pairs(fr.positions, fr.cell, params.rc_radial)
        keep = restrict_pairs(pairs, fr.positions, fr.cell, params.rc_angular)
        tri = triplets_from_pairs(pairs, keep)
        pos_list.append(fr.positions)
        spec_list.append(fr.species)
        pi_list.append(pairs.i + atom_off)
        pj_list.append(pairs.j + atom_off)
        sv_list.append(pairs.shift @ fr.cell)
        keep_list.append(keep + pair_off)
        tij_list.append(tri.pair_ij + pair_off)
        tik_list.append(tri.pair_ik + pair_off)
        soa_list.append(np.full(len(fr.species), s))
        nat_list.append(len(fr.species))
        atom_off += len(fr.species)
        pair_off += len(pairs.i)
    return Graph(
        positions=torch.tensor(np.concatenate(pos_list), dtype=dtype),
        species=torch.tensor(np.concatenate(spec_list), dtype=torch.long),
        pair_i=torch.tensor(np.concatenate(pi_list), dtype=torch.long),
        pair_j=torch.tensor(np.concatenate(pj_list), dtype=torch.long),
        shift_vec=torch.tensor(np.concatenate(sv_list), dtype=dtype),
        ang_keep=torch.tensor(np.concatenate(keep_list), dtype=torch.long),
        tri_ij=torch.tensor(np.concatenate(tij_list), dtype=torch.long),
        tri_ik=torch.tensor(np.concatenate(tik_list), dtype=torch.long),
        struct_of_atom=torch.tensor(np.concatenate(soa_list), dtype=torch.long),
        n_structs=len(nat_list),
        n_atoms_per_struct=torch.tensor(nat_list, dtype=torch.long),
    )


def _cutoff_fn(r: torch.Tensor, rc: float) -> torch.Tensor:
    """Cosine cutoff, zero at and beyond rc, smooth first derivative."""
    inside = r < rc
    fc = 0.5 * (torch.cos(math.pi * r / rc) + 1.0)
    return torch.where(inside, fc, torch.zeros_like(fc))


def compute_descriptors(positions: torch.Tensor, graph: Graph, params: AcsfParams) -> torch.Tensor:
    """Evaluate ACSF descriptors for every atom in the graph.

    Args:
        positions: (N, 3) tensor; pass graph.positions (optionally with
            requires_grad set) so forces can be taken by autograd.
        graph: Packed structure graph.
        params: Symmetry-function parameters.

    Returns:
        (N, n_features) descriptor tensor, differentiable in positions.
    """
    n_atoms = positions.shape[0]
    dtype = positions.dtype
    vec = positions[graph.pair_j] + graph.shift_vec - positions[graph.pair_i]
    r = torch.linalg.norm(vec, dim=1)  # (P,)
    spec_j = graph.species[graph.pair_j]

    # --- radial G2, element-resolved ---
    n_rad = len(params.radial)
    rs = torch.tensor([p[0] for p in params.radial], dtype=dtype)
    eta = torch.tensor([p[1] for p in params.radial], dtype=dtype)
    fc_rad = _cutoff_fn(r, params.rc_radial)
    g2 = torch.exp(-eta[None, :] * (r[:, None] - rs[None, :]) ** 2) * fc_rad[:, None]
    rad_row = graph.pair_i * N_ELEMENTS + spec_j
    radial_out = torch.zeros(n_atoms * N_ELEMENTS, n_rad, dtype=dtype).index_add(0, rad_row, g2)

    # --- angular G4, element-pair-resolved ---
    n_ang = len(params.angular)
    ang_out = torch.zeros(n_atoms * N_PAIR_TYPES, n_ang, dtype=dtype)
    if len(graph.tri_ij) > 0:
        v_ij = vec[graph.tri_ij]
        v_ik = vec[graph.tri_ik]
        r_ij = r[graph.tri_ij]
        r_ik = r[graph.tri_ik]
        v_jk = v_ik - v_ij
        r_jk2 = (v_jk * v_jk).sum(dim=1)
        r_jk = torch.sqrt(torch.clamp(r_jk2, min=1e-12))
        cos_t = (v_ij * v_ik).sum(dim=1) / (r_ij * r_ik)
        cos_t = torch.clamp(cos_t, -1.0, 1.0)
        rca = params.rc_angular
        fc3 = _cutoff_fn(r_ij, rca) * _cutoff_fn(r_ik, rca) * _cutoff_fn(r_jk, rca)
        r2sum = r_ij * r_ij + r_ik * r_ik + r_jk2
        centers = graph.pair_i[graph.tri_ij]
        t1 = spec_j[graph.tri_ij]
        t2 = spec_j[graph.tri_ik]
        pair_type = torch.tensor(_PAIR_INDEX, dtype=torch.long)[t1, t2]
        feats = []
        for eta_a, zeta, lam in params.angular:
            ang = (2.0 ** (1.0 - zeta)) * (1.0 + lam * cos_t) ** zeta
            feats.append(ang * torch.exp(-eta_a * r2sum) * fc3)
        g4 = torch.stack(feats, dim=1)  # (T, n_ang)
        ang_row = centers * N_PAIR_TYPES + pair_type
        ang_out = ang_out.index_add(0, ang_row, g4)

    return torch.cat([radial_out.reshape(n_atoms, -1), ang_out.reshape(n_atoms, -1)], dim=1)
