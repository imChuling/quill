"""集中配置加载：config.yaml 是所有参数的单一事实来源。

用法::

    from quill_config import CFG
    sr = CFG["audio"]["sr"]

`synth/`、`runtime/` 不在实时回调里读配置，而是在加载阶段把需要的标量取出后传参。
"""

from __future__ import annotations

import functools
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


@functools.lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> dict:
    """读取并缓存 config.yaml。"""
    p = Path(path) if path is not None else _CONFIG_PATH
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


CFG = load_config()
