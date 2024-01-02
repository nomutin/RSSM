"""Download robot data from Google Drive."""

from __future__ import annotations

import tarfile
from pathlib import Path

import click
import gdown


@click.command()
@click.argument("url", type=str)
def main(url: str) -> None:
    """Download the data specified in `data_names`."""
    filename = gdown.download(url, quiet=False, fuzzy=True)
    tarfile.open(filename, "r:gz").extractall()
    Path(filename).unlink(missing_ok=False)


if __name__ == "__main__":
    main()