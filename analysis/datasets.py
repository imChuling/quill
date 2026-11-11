"""数据集 loader：IDMT-SMT-Guitar 的 XML 标注 -> 本项目 events 结构（带 label）。

数据集需手动下载（多 GB）；数据缺失时返回空列表。IDMT expression/excitation 风格到
本项目 5 类标签的映射见 `_MAP_EXPRESSION` / `_MAP_EXCITATION`。
"""

from __future__ import annotations

import glob
import os
import xml.etree.ElementTree as ET

# IDMT-SMT-Guitar 的 2 字母码 -> 本项目演奏法标签。
# IDMT 逐音标注只有拨弦风格 + 表情风格，无 legato/strum；IDMT 评估实为
# pluck / slide / sustain(+vibrato) 三类，且重度不平衡，以宏 F1 为准。
_MAP_EXPRESSION = {  # expressionStyle
    "SL": "slide",
    "BE": "slide",  # slide / bending（音高变化）-> slide
    "VI": "sustain",  # vibrato -> sustain(+vibrato)
    "NO": None,
    "I": None,  # normal/不定 -> 由 excitation 定
    "HA": "pluck",
    "DN": "pluck",  # harmonics / dead-note -> pluck
}
_MAP_EXCITATION = {  # excitationStyle（拨弦风格，皆为 pluck 类）
    "PK": "pluck",
    "FS": "pluck",
    "MU": "pluck",
}


def available(root) -> bool:
    return bool(root) and os.path.isdir(root)


def _parse_annotation_xml(path):
    """解析单个 IDMT 标注 XML -> events 列表（容错字段名大小写/层级）。"""
    try:
        tree = ET.parse(path)
    except Exception:
        return []
    root = tree.getroot()
    events = []
    for i, ev in enumerate(root.iter("event")):

        def _get(tag, _ev=ev):
            el = _ev.find(tag)
            return el.text.strip() if el is not None and el.text else None

        onset = _get("onsetSec") or _get("onset")
        offset = _get("offsetSec") or _get("offset")
        pitch = _get("pitch")
        if onset is None or pitch is None:
            continue
        expr = (_get("expressionStyle") or "").strip().upper()  # 码为大写
        exc = (_get("excitationStyle") or "").strip().upper()
        label = _MAP_EXPRESSION.get(expr)
        if label is None:
            label = _MAP_EXCITATION.get(exc, "pluck")  # 拨弦类默认 pluck
        t_on = float(onset)
        t_off = float(offset) if offset else t_on + 0.3
        events.append(
            {
                "id": i,
                "t_on": t_on,
                "t_off": t_off,
                "pitch_midi": float(pitch),
                "velocity_est": 0.0,
                "f0_track": [],
                "features": {},
                "label": label,
                "vibrato": {"rate_hz": 5.5, "depth_cents": 30.0} if expr == "VI" else None,
                "_source_file": os.path.basename(path),
            }
        )
    return events


def load_idmt_smt_guitar(root, limit_files=None):
    """遍历 IDMT-SMT-Guitar 标注目录 -> 统一 events 列表（缺失返回 []）。"""
    if not available(root):
        return []
    xmls = sorted(glob.glob(os.path.join(root, "**", "*.xml"), recursive=True))
    if limit_files:
        xmls = xmls[:limit_files]
    out = []
    for x in xmls:
        out.extend(_parse_annotation_xml(x))
    return out
