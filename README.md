# RSSM

![python](https://img.shields.io/badge/python-3.8-blue)
[![Rye](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/mitsuhiko/rye/main/artwork/badge.json)](https://rye-up.com)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![black](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/psf/black)
[![mypy](https://img.shields.io/badge/mypy-checked-blue)](http://mypy-lang.org/)

Project for RSSM pre-training.

## References

### Paper

- [Dream to Control: Learning Behaviors by Latent Imagination [Hafner+ 2019]](https://arxiv.org/abs/1912.01603)
- [Mastering Atari with Discrete World Models [Hafner+ 2021]](https://arxiv.org/abs/2010.02193)
- Hierarchical Latent Dynamics Model with Multiple Timescales for Learning Long-Horizon Tasks [Fujii+ 2023]

### Code

- [google-research/dreamer](https://github.com/google-research/dreamer)
- [danijar/dreamerv2](https://github.com/danijar/dreamerv2)
- [julisforest/dreamer-pytorch](https://github.com/juliusfrost/dreamer-pytorch)
- [pytorch/rl/dreamer](https://github.com/pytorch/rl/blob/main/examples/dreamer/dreamer.py)

## API

```python
from rssm import RSSMV1

reference = <wandb reference>
rssm = RSSMV1.load_from_wandb(reference=reference)
```

## Dependencies

Modules specified in `pyproject.toml`;

- [einops](https://github.com/arogozhnikov/einops.git)
- [wandb](https://github.com/wandb/wandb.git)
- [nomutin/distribution-extention](https://github.com/nomutin/distribution-extention.git)
- [nomutin/cnn](https://github.com/nomutin/cnn.git)

Modules not specified in `pyproject.toml` (to avoid device errors);

- [torch](https://github.com/pytorch/pytorch.git)
- [lightning](https://github.com/Lightning-AI/pytorch-lightning.git)

## Installation

### pip

```shell
pip install git+https://github.com/nomutin/RSSM.git
```

### poetry

```shell
poetry add git+https://github.com/nomutin/RSSM.git
```

### rye

```shell
rye add rssm --git=https://github.com/nomutin/RSSM.git
```
