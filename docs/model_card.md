# Model card: bpnn_main.pt

## Summary

A Behler-Parrinello neural network potential for BCC Ti-Zr-Nb solid solutions,
implemented from scratch in PyTorch (descriptors, model, training loop) and
distilled from a pretrained universal potential. Committed alongside it:
`linear_main.pt` (ridge baseline on identical descriptors) and
`morse_main.pt` (pairwise baseline), both trained with the same loss, data,
and budget for the benchmark comparison.

## Architecture

- Descriptors: 48 atom-centered symmetry functions per atom; 24 radial G2
  (8 shifted Gaussians x 3 neighbor elements, cutoff 5.5 A) and 24 angular G4
  (4 (eta, zeta, lambda) sets x 6 element pairs, cutoff 4.5 A). Cosine cutoff.
- Per-element feedforward networks (Ti, Zr, Nb), hidden layers (32, 32),
  SiLU activations, float64. Atomic energies sum to the total energy on top of
  a fixed composition-linear reference fit on the training split.
- Forces are exact negative gradients of the energy via autograd; no separate
  force head.
- Parameters: 7,971 (BPNN); linear baseline 147; Morse 21.
- Training: weighted per-atom-energy plus force MSE (force weight 0.1, chosen
  by the committed sweep), Adam, cosine decay, 80 epochs, batch 24, seed 0;
  140 s on CPU.

## Training data (surrogate labels, not DFT)

- 1,566 frames of BCC Ti-Zr-Nb supercells (16 and 54 atoms) across nine
  compositions (seven trained, two transfer-only); rattle, strain, and
  teacher-driven MD batches; 72 generation groups. The main split is 882
  train / 144 val / 192 test / 348 transfer frames, stratified by batch type
  so the test set represents every difficulty class. See `data/README.md`
  and `data/provenance.json`.
- Labels: chgnet 0.4.2 (checkpoint CHGNet v0.3.0, MPtrj-trained),
  BSD-3-Clause, evaluated on CPU at 0.07 s/frame. These are surrogate model
  labels. Nothing here is DFT.
- Split: by generation group and by composition (Ti2ZrNb and TiZrNb2 never
  trained on). Random-frame splitting is used only as a reported control.

## Intended use

- Reproducing and extending the repository benchmark.
- Energy/force prediction, EOS scans, and short MD for BCC Ti-Zr-Nb solid
  solutions near the sampled regime (rattles to ~0.18 A, strains to ~6%,
  MD snapshots to 1400 K, 16 to 54 atom cells).

## Out-of-scope use

- Anything quantitative against experiment: the ceiling of this model is the
  teacher, and the teacher is itself a surrogate for DFT.
- Non-BCC phases (HCP ground states of Ti and Zr, omega phase), surfaces,
  defects, melts: never sampled in training.
- Compositions far outside the Ti-Zr-Nb ternary or cells under strain beyond
  the sampled range.
- Long MD at high temperature; see the measured NVE drift in RESULTS.md before
  trusting trajectories.

## Performance (measured this session, fresh venv, CPU)

Against held-out generation groups (192 frames, stratified over batch
types): energy MAE 8.04 meV/atom, force MAE 50.6 meV/A. Against
compositions never trained on (348 frames): 11.31 meV/atom and 65.1 meV/A
(the linear baseline extrapolates energies slightly better, 9.55, while the
BPNN keeps a 1.6x force lead). Lattice constants from Birch-Murnaghan fits
track the teacher within 0.022 A for Ti and 0.002 A for Nb; BCC Zr is the
weak spot at 0.047 A high and 22 GPa soft. NVE total-energy drift is at or
below 0.001 meV/atom/ps at 300 to 900 K with a 2 fs timestep. Full numbers,
baselines, the split-protocol control (a random-frame split overstates
energy accuracy 2.8x), and the force-weight sweep: `results/metrics.json`
and `RESULTS.md`.

## Provenance and license

- All model code is original (this repository), MIT.
- Teacher: CHGNet (BSD-3-Clause), pip-installed, used for labels only; no
  CHGNet code is vendored.
- DFT anchors: Materials Project, CC BY 4.0.
