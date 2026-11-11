"""Central configuration loading: config.yaml is the single source of truth for all parameters.

Usage::

    from quill_config import CFG
    sr = CFG["audio"]["sr"]

`synth/` and `runtime/` do not read the config in the real-time callback; instead the needed scalars are taken out at load time and passed as arguments.
"""

from __future__ import annotations

import functools
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


@functools.lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> dict:
    """Read and cache config.yaml."""
    p = Path(path) if path is not None else _CONFIG_PATH
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


CFG = load_config()
