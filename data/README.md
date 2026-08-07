# Data provenance

## What the labels are, and are not

Every energy and force label in this project is a **surrogate model label**: it
comes from CHGNet, a pretrained universal interatomic potential installed from
pip, evaluated on CPU. CHGNet was itself trained on DFT data (the MPtrj
dataset), but nothing in this repository is a DFT calculation, and no number
derived from these labels should be quoted as DFT. The exact teacher is
recorded in `provenance.json` (package, version, checkpoint, license), written
at generation time.

Real DFT enters this project in exactly one place: the validation anchors used
by the equation-of-state benchmark, see "Anchors" below.

## Files

- `sample_frames.extxyz` - committed sample: a stratified slice of the full
  dataset (a few frames from every generation group), small enough to commit,
  enough to run the demo and tests against. Carved by
  `scripts/make_sample_data.py`; this is generated data, not a public dataset.
- `provenance.json` - teacher identity, frame and group counts, timings.
- `anchors_literature.json` - committed DFT anchor values with citations
  (Materials Project entries read from the public website, CC BY 4.0, plus
  standard experimental values for Nb).
- `anchors_mp.json` - written by `scripts/fetch_mp_anchors.py` when a free
  Materials Project API key is configured in `MP_API_KEY`; preferred over the
  literature file when present. MP API data is CC BY 4.0.
- `full/dataset.extxyz` - the full labeled dataset, gitignored. Regenerate with
  `python scripts/generate_data.py` (needs the `teacher` extra; about 8 minutes
  on CPU, see the timings in `provenance.json`).

## How the full dataset is built

BCC supercells (16 and 54 atoms) with random site occupancy across nine
compositions: elemental Ti, Zr, Nb; binaries TiZr, TiNb, ZrNb; equiatomic
TiZrNb; and two off-equiatomic compositions (Ti2ZrNb, TiZrNb2) that are never
trained on and serve as the transfer test. Per composition: rattle batches at
three amplitudes, a strain batch (volumetric to +/-6% and shear), and two short
teacher-driven Langevin MD runs (600 K and 1400 K) whose snapshots are labeled.

Every batch carries a generation group id (for example `TiNb_rattle10_b1`).
The benchmark split assigns whole groups, never individual frames, to train,
validation, or test, and holds out the two off-equiatomic compositions
entirely. This matters: sibling frames from one rattle or MD batch are
near-duplicates, and a random frame split quietly leaks them across the split.
The measured size of that effect is reported in `results/split_gap.json` and in
the README.

## Attribution

- Teacher labels: CHGNet (BSD-3-Clause), https://github.com/CederGroupHub/chgnet
- DFT anchors: Materials Project (CC BY 4.0), https://materialsproject.org
- Structure handling and MD driver for generation: ASE (LGPL-2.1+),
  https://wiki.fysik.dtu.dk/ase/

No course-provided, licensed, or third-party proprietary data is used anywhere.
