"""descpot: a from-scratch Behler-Parrinello neural network potential for BCC TiZrNb.

All descriptor, model, and training code in this package is original. Pretrained
universal potentials (the teacher), ASE, and pymatgen are external pip dependencies
used only for labeling, benchmarking, and structure handling.
"""

from descpot.data import Frame, load_frames, save_frames
from descpot.descriptors import AcsfParams, compute_descriptors
from descpot.model import BpnnPotential, LinearPotential, MorsePotential
from descpot.neighbors import neighbor_pairs, triplets_from_pairs

__version__ = "0.1.1"

ELEMENTS = ("Ti", "Zr", "Nb")

__all__ = [
    "AcsfParams",
    "BpnnPotential",
    "ELEMENTS",
    "Frame",
    "LinearPotential",
    "MorsePotential",
    "compute_descriptors",
    "load_frames",
    "neighbor_pairs",
    "save_frames",
    "triplets_from_pairs",
]
