"""Quill 合成端。

引擎铁律 #12：本包只依赖 numpy（+ 后续 numba），不得 import librosa / torch / UI，
这是离线渲染、单测与未来插件移植的共同前提。
"""
