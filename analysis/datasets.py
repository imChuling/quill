"""Dataset loader: IDMT-SMT-Guitar XML annotations -> this project's events structure (with label).

The dataset must be downloaded manually (several GB); returns an empty list if data is missing. The mapping from IDMT expression/excitation
styles to this project's 5 labels is in `_MAP_EXPRESSION` / `_MAP_EXCITATION`.
"""

from __future__ import annotations

import glob
import os
import xml.etree.ElementTree as ET

# IDMT-SMT-Guitar 2-letter codes -> this project's articulation labels.
# IDMT per-note annotations only have plucking style + expression style, no legato/strum; IDMT evaluation is effectively
# three classes pluck / slide / sustain(+vibrato), heavily imbalanced, so macro F1 is the reference.
_MAP_EXPRESSION = {  # expressionStyle
    "SL": "slide",
    "BE": "slide",  # slide / bending (pitch change) -> slide
    "VI": "sustain",  # vibrato -> sustain(+vibrato)
    "NO": None,
    "I": None,  # normal/undetermined -> decided by excitation
    "HA": "pluck",
    "DN": "pluck",  # harmonics / dead-note -> pluck
}
_MAP_EXCITATION = {  # excitationStyle (plucking style, all in the pluck class)
    "PK": "pluck",
    "FS": "pluck",
    "MU": "pluck",
}


def available(root) -> bool:
    return bool(root) and os.path.isdir(root)


def _parse_annotation_xml(path):
    """Parse a single IDMT annotation XML -> events list (tolerant of field-name case/nesting)."""
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
        expr = (_get("expressionStyle") or "").strip().upper()  # codes are upper case
        exc = (_get("excitationStyle") or "").strip().upper()
        label = _MAP_EXPRESSION.get(expr)
        if label is None:
            label = _MAP_EXCITATION.get(exc, "pluck")  # plucked styles default to pluck
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
    """Walk the IDMT-SMT-Guitar annotation directory -> unified events list ([] if missing)."""
    if not available(root):
        return []
    xmls = sorted(glob.glob(os.path.join(root, "**", "*.xml"), recursive=True))
    if limit_files:
        xmls = xmls[:limit_files]
    out = []
    for x in xmls:
        out.extend(_parse_annotation_xml(x))
    return out
