"""tplib v2 行为轨迹单测:受控合成信号 → 提取值必须匹配已知真值。"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.tplib import (  # noqa: E402
    _TRAJ_KEYS,
    FORMAT_VERSION,
    build_library,
    event_trajectories,
    load_tplib,
    save_tplib,
)

SR = 44100


def _tone(f0_curve, amp_curve, dur=1.0, sr=SR):
    """f0(t)/amp(t) 曲线 → 合成正弦(相位积分,无跳变)。"""
    n = int(dur * sr)
    t = np.arange(n) / sr
    f0 = np.interp(t, np.linspace(0, dur, len(f0_curve)), f0_curve)
    amp = np.interp(t, np.linspace(0, dur, len(amp_curve)), amp_curve)
    ph = 2 * np.pi * np.cumsum(f0) / sr
    return (amp * np.sin(ph)).astype(np.float32)


def test_pitch_trajectory_matches_ground_truth():
    """+200 cents 线性上滑 → pitch 轨迹应从约 -100c 到 +100c(残差=减中位)。"""
    y = _tone([220.0, 220.0 * 2 ** (200 / 1200)], [0.5, 0.5], dur=1.0)
    tr = event_trajectories(y, SR, 0.0, 1.0)
    p = tr["pitch"]
    assert p["provenance"] == "extracted" and p["confidence"] > 0.8
    assert p["pts"][2] < -50 and p["pts"][-3] > 50  # 上升斜线
    assert abs(p["pts"][-3] - p["pts"][2] - 200) < 60  # 总幅≈200c


def test_energy_trajectory_swell():
    """幅度从 0.1 渐强到 0.8 → energy 轨迹单调升,总幅≈ln(8)。"""
    y = _tone([330.0, 330.0], [0.1, 0.8], dur=1.0)
    tr = event_trajectories(y, SR, 0.0, 1.0)
    e = tr["energy"]["pts"]
    assert tr["energy"]["confidence"] == 1.0
    assert e[-3] > e[2]
    assert abs((e[-3] - e[2]) - np.log(8)) < 0.5


def test_flat_note_trajectories_near_zero():
    """恒定音 → 四条轨迹都应接近零(残差表示)。"""
    y = _tone([440.0, 440.0], [0.5, 0.5], dur=0.8)
    tr = event_trajectories(y, SR, 0.0, 0.8)
    for k in _TRAJ_KEYS:
        assert np.abs(tr[k]["pts"]).max() < 25, k  # cents/低单位下的静线


def test_short_segment_falls_back_inferred():
    y = _tone([440, 440], [0.5, 0.5], dur=0.02)
    tr = event_trajectories(y, SR, 0.0, 0.02)
    assert all(tr[k]["provenance"] == "inferred" and tr[k]["confidence"] == 0.0 for k in _TRAJ_KEYS)


def test_roundtrip_and_v1_compat(tmp_path):
    y = _tone([220, 260], [0.3, 0.6], dur=1.2)
    events = [
        {"t_on": 0.0, "t_off": 0.55, "label": "pluck", "velocity_est": 0.7},
        {"t_on": 0.6, "t_off": 1.15, "label": "slide", "f0_track": [(0.6, 220.0), (1.1, 300.0)]},
    ]
    lib = build_library(events, y, SR)
    assert lib["meta"]["format_version"] == FORMAT_VERSION == 2
    assert set(lib["pluck"][0]["traj"].keys()) == set(_TRAJ_KEYS)

    p = save_tplib(tmp_path / "t.tplib", lib)
    back = load_tplib(p)
    for k in _TRAJ_KEYS:  # 往返无损
        a, b = lib["pluck"][0]["traj"][k], back["pluck"][0]["traj"][k]
        assert np.allclose(a["pts"], b["pts"]) and a["provenance"] == b["provenance"]
        assert b["confidence"] == pytest.approx(a["confidence"], abs=1e-3)

    # v1 兼容:manifest 无 traj 字段 → 加载 traj=None,不炸
    import json
    import zipfile

    with zipfile.ZipFile(p) as z:
        man = json.loads(z.read("manifest.json"))
        arr = z.read("arrays.npz")
    for sec in ("pluck", "slide", "legato", "strum"):
        for rec in man[sec]:
            rec.pop("traj", None)
    p1 = tmp_path / "v1.tplib"
    with zipfile.ZipFile(p1, "w") as z:
        z.writestr("manifest.json", json.dumps(man))
        z.writestr("arrays.npz", arr)
    old = load_tplib(p1)
    assert old["pluck"][0]["traj"] is None
