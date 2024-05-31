# RSSM

![python](https://img.shields.io/badge/python-3.10-blue)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[RSSMs](https://danijar.com/project/dreamer/) for imitation learning.

## API

```python
from rssm import RSSMV1

reference = "wandb reference"
rssm = RSSMV1.load_from_wandb(reference=reference)
```

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
