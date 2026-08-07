"""Frame container, extxyz persistence, and leakage-safe dataset splits.

A Frame is one labeled periodic structure. Every frame carries a generation
group id (one rattle batch, strain batch, or MD run) and a composition tag,
and the default split operates on those, never on random frames.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SYMBOL_TO_INDEX = {"Ti": 0, "Zr": 1, "Nb": 2}
INDEX_TO_SYMBOL = {v: k for k, v in SYMBOL_TO_INDEX.items()}


@dataclass
class Frame:
    """One periodic structure with optional labels.

    Attributes:
        species: (N,) int array, 0=Ti 1=Zr 2=Nb.
        positions: (N, 3) Cartesian Angstrom.
        cell: (3, 3) row-vector lattice.
        energy: Total energy in eV (surrogate teacher label), or None.
        forces: (N, 3) eV/Angstrom (surrogate teacher label), or None.
        group: Generation-group id, e.g. "TiZrNb_rattle10_b0".
        composition: Composition tag, e.g. "TiZrNb".
    """

    species: np.ndarray
    positions: np.ndarray
    cell: np.ndarray
    energy: float | None = None
    forces: np.ndarray | None = None
    group: str = ""
    composition: str = ""

    @property
    def n_atoms(self) -> int:
        return len(self.species)

    def counts(self) -> np.ndarray:
        """Per-element atom counts, shape (3,)."""
        return np.bincount(self.species, minlength=3)


def save_frames(frames: list[Frame], path: str | Path) -> None:
    """Write frames to an extxyz file (via ASE, structure handling only)."""
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import write

    images = []
    for fr in frames:
        atoms = Atoms(
            symbols=[INDEX_TO_SYMBOL[int(s)] for s in fr.species],
            positions=fr.positions,
            cell=fr.cell,
            pbc=True,
        )
        atoms.info["group"] = fr.group
        atoms.info["composition"] = fr.composition
        if fr.energy is not None:
            atoms.calc = SinglePointCalculator(atoms, energy=fr.energy, forces=fr.forces)
        images.append(atoms)
    write(str(path), images, format="extxyz")


def load_frames(path: str | Path) -> list[Frame]:
    """Read frames from an extxyz file written by save_frames."""
    from ase.io import read

    frames = []
    for atoms in read(str(path), index=":", format="extxyz"):
        energy = None
        forces = None
        if atoms.calc is not None:
            try:
                energy = float(atoms.get_potential_energy())
                forces = np.array(atoms.get_forces())
            except Exception:
                energy, forces = None, None
        frames.append(
            Frame(
                species=np.array([SYMBOL_TO_INDEX[s] for s in atoms.get_chemical_symbols()]),
                positions=np.array(atoms.get_positions()),
                cell=np.array(atoms.get_cell()),
                energy=energy,
                forces=forces,
                group=str(atoms.info.get("group", "")),
                composition=str(atoms.info.get("composition", "")),
            )
        )
    return frames


@dataclass
class Split:
    """Index split of a frame list.

    Attributes:
        train: Training indices.
        val: Validation indices (used for tuning and early model selection).
        test: In-distribution test indices.
        transfer: Held-out-composition test indices (empty for random split).
    """

    train: list[int]
    val: list[int]
    test: list[int]
    transfer: list[int] = field(default_factory=list)


def group_kind(group: str, composition: str) -> str:
    """The batch type of a generation group id, e.g. 'rattle18' or 'md1400'.

    Group ids follow '<composition>_<kind>_b<batch>'; the kind is what remains
    after stripping the composition prefix and the batch suffix.
    """
    kind = group
    if composition and kind.startswith(composition + "_"):
        kind = kind[len(composition) + 1 :]
    return re.sub(r"_b\d+$", "", kind)


def group_split(
    frames: list[Frame],
    holdout_compositions: tuple[str, ...] = (),
    val_fraction: float = 0.1,
    test_fraction: float = 0.15,
    seed: int = 0,
) -> Split:
    """Leakage-safe split: whole generation groups, and whole compositions.

    Frames from one rattle batch, strain batch, or MD run share a group id and
    always land on the same side of the split. Frames whose composition is in
    holdout_compositions go entirely to the transfer set. Held-out groups are
    stratified by batch type (rattle amplitude, strain, MD temperature) so the
    test set's difficulty mix matches the population; with only a few dozen
    groups, an unstratified draw can otherwise miss every hard batch type.

    Args:
        frames: Labeled frames.
        holdout_compositions: Composition tags reserved for transfer testing.
        val_fraction: Fraction of in-distribution groups for validation.
        test_fraction: Fraction of in-distribution groups for testing.
        seed: RNG seed for the group shuffle.

    Returns:
        Split with disjoint train/val/test group sets and a transfer set.
    """
    rng = np.random.default_rng(seed)
    transfer = [k for k, fr in enumerate(frames) if fr.composition in holdout_compositions]
    in_dist = [k for k, fr in enumerate(frames) if fr.composition not in holdout_compositions]
    kind_of = {}
    for k in in_dist:
        kind_of[frames[k].group] = group_kind(frames[k].group, frames[k].composition)
    by_kind: dict[str, list[str]] = {}
    for g in sorted(kind_of):
        by_kind.setdefault(kind_of[g], []).append(g)
    test_groups: set[str] = set()
    val_groups: set[str] = set()
    for kind in sorted(by_kind):
        groups = by_kind[kind]
        rng.shuffle(groups)
        n_test = max(1, int(round(test_fraction * len(groups))))
        n_val = max(1, int(round(val_fraction * len(groups))))
        test_groups.update(groups[:n_test])
        val_groups.update(groups[n_test : n_test + n_val])
    split = Split(train=[], val=[], test=[], transfer=transfer)
    for k in in_dist:
        g = frames[k].group
        if g in test_groups:
            split.test.append(k)
        elif g in val_groups:
            split.val.append(k)
        else:
            split.train.append(k)
    return split


def random_frame_split(
    frames: list[Frame],
    holdout_compositions: tuple[str, ...] = (),
    val_fraction: float = 0.1,
    test_fraction: float = 0.15,
    seed: int = 0,
) -> Split:
    """The optimistic split: random frames regardless of generation group.

    Used only once, to show how much a random-frame split overstates accuracy
    when sibling frames from the same rattle or MD batch land in both train
    and test. Never used for reported model quality.
    """
    rng = np.random.default_rng(seed)
    transfer = [k for k, fr in enumerate(frames) if fr.composition in holdout_compositions]
    in_dist = np.array(
        [k for k, fr in enumerate(frames) if fr.composition not in holdout_compositions]
    )
    perm = rng.permutation(len(in_dist))
    n_test = int(round(test_fraction * len(in_dist)))
    n_val = int(round(val_fraction * len(in_dist)))
    test = in_dist[perm[:n_test]].tolist()
    val = in_dist[perm[n_test : n_test + n_val]].tolist()
    train = in_dist[perm[n_test + n_val :]].tolist()
    return Split(train=train, val=val, test=test, transfer=transfer)
