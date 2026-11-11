"""TimbreSnapshot 数据契约与序列化。

存为 `assets/snapshots/*.npz`。只依赖 numpy 和 json，analysis / synth / runtime 均可导入。

字段：
  f0_ref       float
  H_env        float32[60, Th]   谐波幅度包络
  N_env        float32[40, Tn]   mel 带噪声包络
  attack_res   float32[2646]     60 ms @ 44.1 kHz 起音残差
  partial_freqs float32[60] | None  准谐波声源实测部分音频率；None 为严格 k·f0_ref
  vibrato      {"rate_hz","depth_cents"} | None
  meta         {name, source_file, created, sr, format_version, ...}

加载器宽容：缺失字段给默认值、未知键原样保留，前向/后向兼容。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1

# 渲染所需的框架参数，统一进 meta，render 据此还原
_META_DEFAULTS = {
    "name": "untitled",
    "source_file": "",
    "created": "",
    "sr": 44100,
    "format_version": FORMAT_VERSION,
    "harm_hop": 512,  # H_env 的 STFT hop
    "noise_nfft": 1024,  # N_env 的 STFT 规格
    "noise_hop": 256,
}


def make_snapshot(
    f0_ref,
    H_env,
    N_env,
    attack_res,
    *,
    partial_freqs=None,
    vibrato=None,
    noise_centers_hz=None,
    name="untitled",
    source_file="",
    sr=44100,
    **meta_extra,
) -> dict:
    """组装 TimbreSnapshot（dict）。"""
    meta = dict(_META_DEFAULTS)
    meta.update(
        {
            "name": name,
            "source_file": source_file,
            "sr": int(sr),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "format_version": FORMAT_VERSION,
        }
    )
    meta.update(meta_extra)
    return {
        "f0_ref": float(f0_ref),
        "H_env": np.asarray(H_env, dtype=np.float32),
        "N_env": np.asarray(N_env, dtype=np.float32),
        "attack_res": np.asarray(attack_res, dtype=np.float32),
        "partial_freqs": (
            None if partial_freqs is None else np.asarray(partial_freqs, dtype=np.float32)
        ),
        "vibrato": vibrato,
        "noise_centers_hz": (
            None if noise_centers_hz is None else np.asarray(noise_centers_hz, dtype=np.float32)
        ),
        "meta": meta,
    }


def save_snapshot(path, snap: dict) -> Path:
    """序列化到 .npz（往返无损）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    has_partials = snap.get("partial_freqs") is not None
    has_centers = snap.get("noise_centers_hz") is not None
    arrays = {
        "H_env": np.asarray(snap["H_env"], dtype=np.float32),
        "N_env": np.asarray(snap["N_env"], dtype=np.float32),
        "attack_res": np.asarray(snap["attack_res"], dtype=np.float32),
        "f0_ref": np.float64(snap["f0_ref"]),
        "has_partials": np.bool_(has_partials),
        "partial_freqs": (
            np.asarray(snap["partial_freqs"], dtype=np.float32)
            if has_partials
            else np.zeros(0, dtype=np.float32)
        ),
        "noise_centers_hz": (
            np.asarray(snap["noise_centers_hz"], dtype=np.float32)
            if has_centers
            else np.zeros(0, dtype=np.float32)
        ),
        "meta_json": np.array(json.dumps(snap.get("meta", {}), ensure_ascii=False)),
        "vibrato_json": np.array(json.dumps(snap.get("vibrato"))),
    }
    np.savez(path, **arrays)
    if path.suffix != ".npz":  # np.savez 会自动补 .npz
        path = path.with_suffix(".npz")
    return path


def load_snapshot(path) -> dict:
    """从 .npz 反序列化为 dict，缺失字段给默认值（宽容加载）。"""
    data = np.load(path, allow_pickle=False)
    keys = set(data.files)

    def get(k, default=None):
        return data[k] if k in keys else default

    meta = dict(_META_DEFAULTS)
    if "meta_json" in keys:
        try:
            meta.update(json.loads(str(data["meta_json"])))
        except Exception:
            pass

    vibrato = None
    if "vibrato_json" in keys:
        try:
            vibrato = json.loads(str(data["vibrato_json"]))
        except Exception:
            vibrato = None

    has_partials = bool(get("has_partials", np.bool_(False)))
    pf = get("partial_freqs")
    partial_freqs = (
        np.asarray(pf, dtype=np.float32) if has_partials and pf is not None and pf.size else None
    )

    centers = get("noise_centers_hz")
    noise_centers_hz = (
        np.asarray(centers, dtype=np.float32) if centers is not None and centers.size else None
    )

    snap = {
        "f0_ref": float(get("f0_ref", np.float64(440.0))),
        "H_env": np.asarray(get("H_env"), dtype=np.float32),
        "N_env": np.asarray(get("N_env", np.zeros((40, 1))), dtype=np.float32),
        "attack_res": np.asarray(get("attack_res", np.zeros(2646)), dtype=np.float32),
        "partial_freqs": partial_freqs,
        "vibrato": vibrato,
        "noise_centers_hz": noise_centers_hz,
        "meta": meta,
    }
    return snap
