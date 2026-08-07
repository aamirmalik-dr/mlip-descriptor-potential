# descpot Python API

The package is a small library. Everything below is importable from `descpot`
or its submodules; all public functions carry type hints and docstrings.

## Structures and data

```python
from descpot.structures import bcc_supercell, rattle_batch, strain_batch, COMPOSITIONS

frame = bcc_supercell("TiZrNb", reps=2, seed=0)      # 16-atom random solid solution
batch = rattle_batch("TiNb", reps=2, amplitude=0.10, n_frames=30, seed=1, batch=0)
```

A `Frame` (in `descpot.data`) holds `species` (0=Ti, 1=Zr, 2=Nb), `positions`,
`cell`, optional `energy`/`forces` labels, a generation `group` id, and a
`composition` tag. Persistence is plain extxyz:

```python
from descpot.data import save_frames, load_frames

save_frames(frames, "my_frames.extxyz")
frames = load_frames("my_frames.extxyz")
```

Splits never separate frames of one generation batch, and can hold out whole
compositions:

```python
from descpot.data import group_split

split = group_split(frames, holdout_compositions=("Ti2ZrNb", "TiZrNb2"), seed=0)
split.train, split.val, split.test, split.transfer  # index lists
```

## Descriptors

```python
from descpot.descriptors import AcsfParams, build_graph, compute_descriptors

acsf = AcsfParams()                    # 48 features: 24 radial G2 + 24 angular G4
graph = build_graph(frames, acsf)      # packs any number of frames into one graph
desc = compute_descriptors(graph.positions, graph, acsf)   # (n_atoms, 48), differentiable
```

`build_graph` precomputes periodic neighbor pairs and angular triplets with the
from-scratch neighbor list in `descpot.neighbors` (verified against ASE in the
tests). Shift vectors are premultiplied by each frame's own cell, so strained
and unstrained frames batch together.

## Models

All three potentials share one interface:

```python
from descpot.model import BpnnPotential, LinearPotential, MorsePotential

model = BpnnPotential(acsf, hidden=(32, 32))
energies = model(graph.positions, graph)        # (n_structures,) total energies, eV
energies, forces = model.energy_forces(graph)   # forces from autograd, eV/A
```

Checkpoint IO keeps enough metadata to reconstruct the exact model:

```python
from descpot.model import save_potential, load_potential

save_potential(model, "model.pt")
model = load_potential("model.pt")
```

## Training

```python
from descpot.training import TrainSettings, train_potential, tune_linear_ridge

settings = TrainSettings(epochs=80, batch_size=24, lr=3e-3, force_weight=0.1, seed=0)
history = train_potential(model, frames, split.train, split.val, settings)
```

`train_potential` fits the composition-linear reference energies and the
per-element descriptor scaler on the training split only, then optimizes the
weighted per-atom-energy plus force MSE with Adam and cosine decay, keeping the
best validation state. The same function trains the linear and Morse baselines,
which is what makes the benchmark comparison budget-matched.
`tune_linear_ridge` additionally sweeps the ridge strength on validation.

## Evaluation, EOS, MD

```python
from descpot.training import prepare_batches, evaluate_batches
from descpot.eos import ev_curve_model, fit_eos, default_scales
from descpot.md import run_nve

metrics = evaluate_batches(model, prepare_batches(frames, split.test, acsf, 24))
vols, energies = ev_curve_model(model, "Nb", reps=2, scales=default_scales("Nb"))
fit = fit_eos(vols, energies)           # a0, B0 (GPa), B0', E0 via Birch-Murnaghan
result = run_nve(model, frame, temperature_k=600, timestep_fs=2.0, n_steps=1500)
result.drift_mev_per_atom_per_ps
```

## Teacher labeling (optional extra)

```python
from descpot.teacher import get_teacher, label_frames

calc, info = get_teacher("chgnet")     # or "mace-mp0-small"
label_frames(frames, calc)             # surrogate labels in place, NOT DFT
```

## CLI

```
descpot predict --model models/bpnn_main.pt --xyz data/sample_frames.extxyz --forces
descpot eos     --model models/bpnn_main.pt --composition Nb
descpot md      --model models/bpnn_main.pt --composition TiZrNb --steps 500
descpot info    --model models/bpnn_main.pt
```
