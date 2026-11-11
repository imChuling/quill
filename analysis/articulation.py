"""Articulation classification: a rule priority tree labels each event.

Rule priority (all thresholds read from config.yaml):
    onset≥3 in the same window and gap<80ms             -> strum
    |net f0 slope|>threshold and duration>60ms (and not vibrato) -> slide
    joined to previous note, no new strong attack, continuous amplitude, small interval -> legato
    log_attack<25ms and centroid>2kHz                    -> pluck
    otherwise                                            -> sustain; significant 4-8Hz f0 peak -> +vibrato

slide and vibrato are easily confused: first extract periodicity (vibrato field, net slope≈0), then judge monotonicity (large net slope).
A random forest (sklearn) serves as the control classifier for the rule tree.
"""

from __future__ import annotations

import numpy as np

try:
    from quill_config import CFG

    _A = CFG.get("articulation", {})
except Exception:  # pragma: no cover
    _A = {}

_LABELS = ("pluck", "slide", "legato", "strum", "sustain")

_DEF = {
    "strum_window_ms": 80.0,
    "strum_min_onsets": 3,
    "slide_slope_cents_s": 500.0,
    "slide_min_dur_ms": 60.0,
    "slide_monotonic_frac": 0.70,
    "legato_max_gap_ms": 60.0,
    "legato_pitch_jump_max_st": 7.0,
    "pluck_attack_ms": 25.0,
    "pluck_centroid_hz": 2000.0,
    "vibrato_rate_lo": 4.0,
    "vibrato_rate_hi": 8.0,
}


def _f0_monotonic_frac(track):
    """Fraction of same-direction steps in the f0 track = max(up fraction, down fraction). A true slide≈1; octave jumps/noise≈0.5."""
    if len(track) < 3:
        return 0.0
    hz = np.array([p[1] for p in track], dtype=np.float64)
    d = np.diff(hz)
    d = d[np.abs(d) > 1e-6]
    if d.size == 0:
        return 0.0
    up = float(np.mean(d > 0))
    return max(up, 1.0 - up)


def _cfg(key):
    return float(_A.get(key, _DEF[key]))


def _is_vibrato(ev):
    """Whether the event carries 4-8Hz vibrato."""
    v = ev.get("vibrato")
    if not v:
        return False
    r = v.get("rate_hz", 0.0)
    return _cfg("vibrato_rate_lo") <= r <= _cfg("vibrato_rate_hi")


def _strum_windows(events):
    """Return the set of event ids belonging to a strum: onset count in a sliding window≥threshold, gap<window."""
    win = _cfg("strum_window_ms") / 1000.0
    need = int(_A.get("strum_min_onsets", _DEF["strum_min_onsets"]))
    ev = sorted(events, key=lambda e: e["t_on"])
    strum_ids = set()
    n = len(ev)
    for i in range(n):
        j = i
        while j < n and ev[j]["t_on"] - ev[i]["t_on"] < win:
            j += 1
        if j - i >= need:
            for k in range(i, j):
                strum_ids.add(ev[k]["id"])
    return strum_ids


def classify_events(events, prev_aware=True, valid_labels=None):
    """Label each event with the rule priority tree, writing ev["label"] in place (and returning events).

    valid_labels: restricts the output label space. Isolated-note datasets (e.g. IDMT) only have pluck/slide/sustain,
    so strum/legato detection is turned off and pluck is the default, avoiding predictions for classes absent from the dataset.
    None=all 5 classes.
    """
    if not events:
        return events
    valid_labels = set(_LABELS) if valid_labels is None else set(valid_labels)
    use_strum = "strum" in valid_labels
    use_legato = "legato" in valid_labels and prev_aware
    ev_sorted = sorted(events, key=lambda e: e["t_on"])
    strum_ids = _strum_windows(events) if use_strum else set()

    slope_thr = _cfg("slide_slope_cents_s")
    slide_min_dur = _cfg("slide_min_dur_ms") / 1000.0
    gap_max = _cfg("legato_max_gap_ms") / 1000.0
    jump_max = _cfg("legato_pitch_jump_max_st")
    atk_thr = np.log10(_cfg("pluck_attack_ms") / 1000.0)
    cen_thr = _cfg("pluck_centroid_hz")

    restricted = not (use_strum or use_legato)  # isolated-note domain: pluck is the default
    prev = None
    for ev in ev_sorted:
        f = ev.get("features", {})
        vib = _is_vibrato(ev)
        slope = abs(float(f.get("f0_slope_cents_s", 0.0) or 0.0))
        dur = ev["t_off"] - ev["t_on"]
        log_atk = f.get("log_attack")
        cen = f.get("centroid_mean")
        mono = _f0_monotonic_frac(ev.get("f0_track", []))
        # slide: large net slope + long enough + monotonic + not vibrato (extract periodicity first, then judge monotonicity)
        is_slide = (
            (not vib)
            and slope > slope_thr
            and dur > slide_min_dur
            and mono >= _cfg("slide_monotonic_frac")
        )

        if use_strum and ev["id"] in strum_ids:
            label = "strum"
        elif is_slide:
            label = "slide"
        elif vib:  # vibrato -> sustain
            label = "sustain"
        elif use_legato and prev is not None and _is_legato(ev, prev, gap_max, jump_max, atk_thr):
            label = "legato"
        elif restricted or (
            log_atk is not None
            and np.isfinite(log_atk)
            and log_atk < atk_thr
            and cen is not None
            and np.isfinite(cen)
            and cen > cen_thr
        ):  # isolated note: neither slide nor vibrato means pluck
            label = "pluck"
        else:
            label = "sustain"

        ev["label"] = label
        prev = ev
    return events


def _is_legato(ev, prev, gap_max, jump_max, atk_thr):
    """legato: joined to the previous note (small gap), interval in range, no strong attack on this note (slow attack), continuous amplitude."""
    gap = ev["t_on"] - prev["t_off"]
    if gap > gap_max:
        return False
    p0, p1 = prev.get("pitch_midi"), ev.get("pitch_midi")
    if p0 is None or p1 is None:
        return False
    # the same pitch does not count as legato; too large a span is more like an independent attack
    if abs(p1 - p0) > jump_max or abs(p1 - p0) < 0.5:
        return False
    log_atk = ev.get("features", {}).get("log_attack")
    soft = (log_atk is None) or (not np.isfinite(log_atk)) or (log_atk >= atk_thr)
    # continuous amplitude: this note's starting level is close to the previous note's ending (no hard hit)
    cont = ev.get("velocity_est", 0.0) <= prev.get("velocity_est", 1.0) * 1.3 + 0.1
    return bool(soft and cont)


# --------------------------------------------------------------------------- #
# Random forest classifier (control for the rule tree)
# --------------------------------------------------------------------------- #
_FEATURE_KEYS = (
    "log_attack",
    "centroid_mean",
    "centroid_slope",
    "f0_slope_cents_s",
    "polyphony",
    "hnr",
)


def feature_vector(ev):
    """Event -> numeric feature vector (forest input)."""
    f = ev.get("features", {})
    vec = []
    for k in _FEATURE_KEYS:
        v = f.get(k)
        vec.append(float(v) if v is not None and np.isfinite(v) else 0.0)
    vec.append(1.0 if _is_vibrato(ev) else 0.0)
    vec.append(float(ev["t_off"] - ev["t_on"]))
    return np.asarray(vec, dtype=np.float64)


def train_forest(events, labels, n_estimators=100, seed=0):
    """Train a random forest on labelled events. Returns an sklearn classifier."""
    from sklearn.ensemble import RandomForestClassifier

    X = np.array([feature_vector(e) for e in events])
    clf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=seed, class_weight="balanced"
    )
    clf.fit(X, labels)
    return clf


def forest_predict(clf, events):
    X = np.array([feature_vector(e) for e in events])
    return list(clf.predict(X))
