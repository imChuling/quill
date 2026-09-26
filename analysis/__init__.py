"""Quill 分析端（离线）。

引擎铁律 #12：librosa / pesto 等重依赖只许活在 `analysis/`，
`synth/` 与 `runtime/` 只依赖 numpy(+numba)。
"""
