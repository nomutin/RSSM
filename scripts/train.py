"""
モデルの学習.

References
----------
- https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced.html

"""

import shutil
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from lightning.pytorch.cli import LightningCLI


@contextmanager
def temporary_directory(path: Path) -> Generator[Path, None, None]:
    """完全パスを指定可能なTemporary Directory."""
    if path.exists():
        msg = f"Path {path} already exists."
        raise FileExistsError(msg)
    path.mkdir(parents=False)

    try:
        yield path
    finally:
        shutil.rmtree(path)


def main() -> None:
    """Execute lightning cli."""
    with temporary_directory(Path("tmp")) as _:
        LightningCLI(save_config_callback=None)


if __name__ == "__main__":
    main()
