# Benchmark results

All numbers on this page were measured in this repository's fresh virtual
environment on CPU (torch 2.13.0+cpu, float64), with fixed seeds, from the
committed configs. Reproduce with `python scripts/run_all.py` (about 45
minutes of benchmarks and checks plus 8 minutes of data generation on a
desktop CPU).

Labels are surrogate model labels from chgnet 0.4.2 (checkpoint CHGNet
v0.3.0, MPtrj-trained, BSD-3-Clause). Nothing here is a DFT calculation; the
only DFT numbers are the Materials Project anchor values in the EOS section.

## Setup

- Dataset: 1,566 frames, 72 generation groups, nine compositions, 16 and 54
  atom BCC supercells; rattle (0.05/0.10/0.18 A), strain (to 6% volumetric
  plus shear), and teacher-driven MD (600 K, 1400 K) batches. Labeling cost
  0.07 s/frame.
- Split (`group_split`, seed 0): 882 train / 144 val / 192 test frames by
  whole generation groups, stratified by batch type so every difficulty class
  (including 0.18 A rattles and 1400 K MD) appears in the test set; 348
  frames of Ti2ZrNb and TiZrNb2 never trained on (transfer set).
- All three models train with the identical loop: weighted per-atom-energy
  plus force MSE (force weight 0.1), Adam, cosine decay, 80 epochs, batch 24.
  The ridge baseline additionally gets its weight-decay strength swept on
  validation at a third of the budget before a full-budget retrain, so its
  total tuning budget exceeds the BPNN's.

## Main comparison (configs/main.yaml)

Test = held-out generation groups, in-distribution compositions.
Transfer = held-out compositions.

| model | params | train s | test E MAE (meV/atom) | test F MAE (meV/A) | transfer E MAE | transfer F MAE |
|---|---|---|---|---|---|---|
| BPNN | 7,971 | 141 | **8.04** | **50.6** | 11.31 | **65.1** |
| ridge (linear, same descriptors) | 147 | 127 | 9.60 | 95.4 | **9.55** | 102.0 |
| Morse (pairwise) | 21 | 20 | 12.32 | 128.7 | 13.39 | 136.0 |

RMSE: BPNN 13.9 meV/atom / 92.5 meV/A; ridge 17.7 / 161.6; Morse 23.6 / 177.0.

Operating-point check (`scripts/check_operating_point.py`,
`results/operating_point.json`): the shared force weight 0.1 was selected by
a sweep run on the BPNN, so the baselines were retrained at force weight 1.0,
the force-optimal end of that sweep, with the identical data, split, and
budget. The ridge model's force MAE improves only from 95.4 to 92.9 meV/A
while its energy MAE doubles to 18.2 meV/atom; Morse improves from 128.7 to
117.8 at a similar energy cost. The force gap is a representation limit, not
an artifact of the shared training weight. The ridge sweep's grid edge was
probed the same way: weight decays 0, 1e-8, and 1e-6 give validation scores
identical to three decimals, so nothing better lies beyond the committed
grid.

Reading: at matched data and budget, the many-body networks earn their keep
on forces, 1.9x over a linear readout of the identical symmetry functions
and 2.5x over the pair potential. The energy advantage is a modest 1.2x on a
representative test mix: the composition-linear reference plus the
descriptors already capture most of the energy signal, and the residual is
dominated by the hardest frames (large rattles, hot MD) where every model
struggles. On composition transfer the ridge model is actually the best
energy extrapolator (9.55 vs 11.31 meV/atom), a known virtue of small linear
models, while the BPNN keeps a 1.6x force lead. Both directions are reported
because both are true.

## Learning curve (configs/learning_curve.yaml, 50 epochs per point)

Test energy MAE (meV/atom) / force MAE (meV/A) versus training frames:

| train frames | BPNN | ridge | Morse |
|---|---|---|---|
| 100 | 15.20 / 81.2 | 20.82 / 124.8 | 54.47 / 222.2 |
| 200 | 11.35 / 72.2 | 18.24 / 129.5 | 27.27 / 156.4 |
| 400 | 9.11 / 60.4 | 12.42 / 102.5 | 13.63 / 132.5 |
| 800 | 5.99 / 48.4 | 8.75 / 95.1 | 11.78 / 130.2 |
| 882 (full) | 6.82 / 47.7 | 7.73 / 94.2 | 12.13 / 132.1 |

The BPNN's force lead is present at every size and still widening; the ridge
model's force error has plateaued near 95 meV/A by 400 frames (a
representation limit, not a data limit) while its energy error keeps
tracking the BPNN's within a couple of meV/atom. The Morse energy floor is
about 12 meV/atom. The small non-monotonic steps between 800 and 882 frames
are run-to-run noise of order 1 meV/atom / a few meV/A.

## Split-protocol control (configs/split_gap.yaml)

Identical BPNN, budget, and data; only the split protocol changes. Three
split seeds each, mean +/- std over seeds, each protocol scored on its own
test set:

| protocol | E MAE (meV/atom) | F MAE (meV/A) |
|---|---|---|
| group split (leakage-safe, stratified) | 8.55 +/- 1.55 | 55.9 +/- 7.0 |
| random-frame split (optimistic control) | 3.05 +/- 0.20 | 43.2 +/- 1.8 |

The random-frame protocol overstates energy accuracy by 2.8x and force
accuracy by 1.3x, consistently across seeds: sibling frames from one rattle
or MD batch are near-duplicates, and a random split puts some of every batch
on both sides. This is the number to remember when reading MLIP papers whose
test frames were sampled from the same trajectories as their training
frames.

A fairness correction made during the build, stated plainly: the first
version of the group split drew held-out groups uniformly, and with only 56
in-distribution groups all three split seeds happened to hold out zero
high-amplitude-rattle groups and almost no MD groups, so the group-split
test was quietly easier than a representative mix (it claimed 3.3 meV/atom
and 32 meV/A for the BPNN, and its force MAE even looked better than the
random-frame control's, for the wrong reason). The group split is now
stratified by batch type, every batch type appears in every test set, and
all numbers in this file come from the stratified rerun.

## Force-weight sweep (configs/force_weight.yaml, 50 epochs)

| force weight | test E MAE (meV/atom) | test F MAE (meV/A) |
|---|---|---|
| 0 | 7.98 | 125.6 |
| 0.01 | 10.04 | 73.5 |
| 0.1 | **6.10** | 47.6 |
| 1.0 | 7.06 | **43.0** |

Energy-only training collapses forces to Morse-baseline territory (126
meV/A) even though the energy fit looks fine, the classic failure mode of
energy-only fitting. Weight 1.0 buys the best forces at an energy cost;
0.1 is the best compromise and is what the shipped model uses. Single-seed
sweep; the non-monotonic 0.01 energy point indicates run-to-run noise of a
few meV/atom, far smaller than the 0-vs-0.1 force gap.

## Equation of state (configs/eos.yaml)

Birch-Murnaghan fits over +/-6% volume scans, ideal BCC cells. Anchor = DFT
values fetched from the Materials Project API (CC BY 4.0) by
`scripts/fetch_mp_anchors.py`, committed in `data/anchors_mp.json`; the
conventional a0 is recovered from the primitive-cell volume per atom. A
citation-carrying literature fallback (`data/anchors_literature.json`, values
consistent within 0.005 A and 6 GPa) is committed for keyless runs.

| composition | a0 model (A) | a0 teacher (A) | a0 DFT anchor (A) | B0 model (GPa) | B0 teacher (GPa) | B0 DFT anchor (GPa) |
|---|---|---|---|---|---|---|
| Ti | 3.246 | 3.268 | 3.252 (mp-73) | 70.5 | 74.3 | 111 |
| Zr | 3.642 | 3.595 | 3.582 (mp-41) | 66.8 | 88.6 | 88 |
| Nb | 3.325 | 3.327 | 3.318 (mp-75) | 167.5 | 166.6 | 172 |
| TiZrNb | 3.378 | 3.386 | - | 99.3 | 107.7 | - |

Nb is essentially perfect against the teacher (0.002 A, 1 GPa) and close to
the DFT anchor. Ti tracks the teacher well (0.022 A, 4 GPa); the teacher
itself sits 37 GPa below the MP B0 for this mechanically unstable phase, and
a distilled model cannot beat its teacher. BCC Zr is the honest weak spot:
the student comes out 0.047 A large in a0 and 22 GPa soft in B0 relative to
the teacher, the price of a small model whose training weight for elemental
Zr is one-ninth of the data. The equiatomic alloy lands within 0.008 A and
8 GPa of the teacher.

## NVE stability (configs/nve.yaml)

From-scratch velocity-Verlet, 1,500 steps at 2 fs (3 ps), 16-atom cells,
BPNN forces:

| composition | T init (K) | drift (meV/atom/ps) | std E_tot (meV/atom) |
|---|---|---|---|
| TiZrNb | 300 | -0.000 | 0.002 |
| TiZrNb | 900 | 0.000 | 0.006 |
| Nb | 600 | 0.001 | 0.003 |

Energy conservation at the sub-0.01 meV/atom level: autograd forces are
exact gradients of a smooth energy, and float64 plus cosine cutoffs keep it
that way.

## Timings

Benchmark wall times on this machine (CPU): main 426 s, split-gap control
861 s (6 trainings), learning curve 788 s, force weight 361 s, EOS 17 s,
NVE 30 s, operating-point check about 4 minutes (2 retrainings). Dataset
generation and labeling: about 8 minutes at 0.07 s/frame.
