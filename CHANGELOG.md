# Changelog

## Unreleased

- Package metadata completed: keywords, classifiers, and project URLs in `pyproject.toml`; citation file added.

## 0.1.0 (2026-08-07)

- First public release: from-scratch Behler-Parrinello neural network potential for BCC TiZrNb with original periodic neighbor lists, atom-centered symmetry functions, per-element networks, autograd forces, and the training loop in plain PyTorch.
- Benchmark against tuned linear and pair-potential baselines at matched training data and budget, with splits by generation group and composition and the random-frame split gap shown once.
- Equation-of-state benchmark against Materials Project anchors fetched through the API, with the literature fallback recorded.
- Operating-point fairness check for the force-error headline, so the comparison is made at each baseline's best setting rather than at one shared setting.
- Force learning curve placed on the hero figure; figure axes cleaned up.
- Documentation synchronised with the committed metrics (EOS and training wall times); the reference-fit warning is asserted in the tests.
- Executed tutorial notebook with cleaned imports, linted in CI.
