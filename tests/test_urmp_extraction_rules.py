"""The rules in tools/extract_trajectories_urmp.py must reproduce the shipped URMP
libraries exactly: articulation labels, phrase flags, gap symmetry, duration filter.

This is the evidence that the restored script is the one that built the libraries.
Needs the (non-redistributed) library files; skipped when they are absent.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from extract_trajectories_urmp import (  # noqa: E402
    MIN_NOTE_DUR,
    PHRASE_GAP,
    classify_articulation,
    context_features,
)

LIBS = {
    "vn": ROOT / "checkpoints" / "urmp_vn_trajectories.npz",
    "tpt": ROOT / "checkpoints" / "urmp_tpt_trajectories.npz",
}


def _load(inst):
    p = LIBS[inst]
    if not p.exists():
        pytest.skip(f"{p.name} not present (URMP-derived, not redistributed)")
    d = np.load(p, allow_pickle=True)
    return d["features"], d["trajectories"], d["confidence"], json.loads(str(d["meta"]))


@pytest.mark.parametrize("inst", sorted(LIBS))
def test_classifier_reproduces_every_stored_label(inst):
    F, T, C, meta = _load(inst)
    pred = np.array([classify_articulation(T[i, 0], float(C[i, 0])) for i in range(len(F))])
    assert np.array_equal(pred, F[:, 6].astype(int)), int((pred != F[:, 6]).sum())
    assert np.array_equal(pred, [m["art"] for m in meta])


@pytest.mark.parametrize("inst", sorted(LIBS))
def test_context_rules_match_stored_features(inst):
    F, _, _, meta = _load(inst)
    assert F[:, 1].min() >= MIN_NOTE_DUR
    assert np.allclose(F[:, 0], [m["midi"] for m in meta])
    note_idx = np.array([m["note_idx"] for m in meta])
    first_rule = ((note_idx == 0) | (F[:, 2] > PHRASE_GAP)).astype(np.float32)
    assert np.array_equal(first_rule, F[:, 4])
    # consecutive kept notes of a track: my next_gap == your prev_gap
    track = np.array([m["track"] for m in meta])
    for t in np.unique(track):
        idx = np.where(track == t)[0]
        for a, b in zip(idx[:-1], idx[1:]):
            if note_idx[b] == note_idx[a] + 1:
                assert abs(F[a, 3] - F[b, 2]) < 1e-5


def test_context_features_on_synthetic_phrase():
    """Two phrases separated by a long rest; gap/phrase flags follow the stored convention."""
    notes = [(0.0, 0.5, 60.0), (0.5, 0.5, 62.0), (1.6, 0.4, 64.0)]  # rest 0.6 s before note 3
    f0 = context_features(notes, 0, 0)
    f1 = context_features(notes, 1, 1)
    f2 = context_features(notes, 2, 2)
    assert f0.tolist() == pytest.approx([60.0, 0.5, 1.0, 0.0, 1.0, 0.0, 0.0])
    assert f1[2] == pytest.approx(0.0) and f1[3] == pytest.approx(0.6) and f1[5] == 1.0 and f1[6] == 1.0
    assert f2[2] == pytest.approx(0.6) and f2[3] == 1.0 and f2[4] == 1.0 and f2[5] == 1.0
