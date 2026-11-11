"""MIDI dialect adapter layer: any dialect in, one unified IR out.

There is only one SMF container, but articulation information hides in different places depending on instrument-community conventions (dialects).
This module detects the dialect automatically and converges to Quill's unified intermediate representation:

    [{"midi": int, "dur": float, "art": int, "onset": float}, ...]
    art: 0=plain 1=vibrato 2=slide (same as neural/perform.py)

Design constraints:
- this module only produces a score and does not import neural.*
- automatic detection first; on low confidence the report lists candidates for the UI to decide
- drums are an explicit non-goal: a pure drum file is rejected outright

Dialects and adaptation actions:
  piano         pass-through (velocity/CC64 do not carry information Quill needs)
  keyswitch     remove out-of-range trigger notes, map latching switches to art labels
  multichannel  guitar/MPE: merge per instrument track, convert bend curves to art labels
  mono_lead     monophonic + bend: convert bend curves to art labels
  breath        continuous CC2 curve: extract per-note energy hints (cc_energy, extra key ignored by the engine)
  drums         rejected (the pitched monophonic pipeline does not apply)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---- Constants ----
BEND_RANGE_ST = 2.0  # standard GM pitch-bend range ±2 semitones (can be overridden by the file, often missing in SMF)
KS_GAP_ST = 12  # minimum gap (semitones) between the keyswitch cluster and the playing range
KS_MAX_DUR = 0.30  # maximum keyswitch note duration (seconds)
SLIDE_MIN_ST = 1.0  # minimum net displacement (semitones) for a bend to count as a slide
SLIDE_MONO_FRAC = 0.8  # monotonic step fraction for a bend to count as a slide
VIB_RATE_LO, VIB_RATE_HI = 3.0, 12.0  # frequency window (Hz) for a bend to count as vibrato
VIB_MIN_AMP_ST = 0.08  # minimum peak amplitude (semitones, ≈8 cents) for a bend to count as vibrato
CHORD_EPS = 0.010  # time tolerance (seconds) for treating notes at the same onset as a chord

# keyswitch offset within the cluster -> art label (can be overridden by the caller)
DEFAULT_KS_MAP = {0: 0, 1: 1, 2: 2}  # cluster bottom=plain, +1=vibrato, +2=slide

ART_NAMES = {0: "plain", 1: "vibrato", 2: "slide"}


@dataclass
class ImportReport:
    dialect: str = "piano"
    confidence: float = 1.0
    candidates: list = field(default_factory=list)  # alternative dialects on low confidence
    n_notes_in: int = 0
    n_notes_out: int = 0
    n_channels_merged: int = 0
    n_keyswitches_stripped: int = 0
    ks_map_used: dict = field(default_factory=dict)  # trigger pitch -> art name
    n_bend_slide: int = 0
    n_bend_vibrato: int = 0
    n_overlaps_trimmed: int = 0
    n_chord_notes_dropped: int = 0
    n_drum_tracks_dropped: int = 0
    cc_energy_source: str = ""  # "cc1"/"cc2"/"cc11"/""
    warnings: list = field(default_factory=list)


class DrumFileError(ValueError):
    """Pure drum file: note numbers are instrument identities, not pitches; the pipeline explicitly does not support this."""


# ---- Detection ----


def _melodic_instruments(pm):
    return [ins for ins in pm.instruments if not ins.is_drum and ins.notes]


def _find_keyswitch_cluster(notes):
    """Find a keyswitch cluster in a single-instrument note set.

    Criterion: after sorting by pitch there is a gap of ≥KS_GAP_ST semitones, and all notes below the gap
    are short (≤KS_MAX_DUR). Returns (set of pitches in the cluster, cluster bottom pitch) or (set(), None).
    """
    if len(notes) < 4:
        return set(), None
    pitches = sorted({n.pitch for n in notes})
    for i in range(len(pitches) - 1):
        if pitches[i + 1] - pitches[i] >= KS_GAP_ST:
            low = set(pitches[: i + 1])
            low_notes = [n for n in notes if n.pitch in low]
            main_notes = [n for n in notes if n.pitch not in low]
            if not low_notes or not main_notes:
                continue
            if all(n.end - n.start <= KS_MAX_DUR for n in low_notes) and len(main_notes) >= len(
                low_notes
            ):
                return low, min(low)
    return set(), None


def _cc_density(ins, number):
    ccs = [c for c in ins.control_changes if c.number == number]
    if not ins.notes or len(ccs) < 4:
        return 0.0
    span = max(n.end for n in ins.notes) - min(n.start for n in ins.notes)
    return len(ccs) / max(span, 1e-6)


def detect_dialect(pm):
    """Return (dialect, confidence, candidates, signals_dict)."""
    melodic = _melodic_instruments(pm)
    drums = [ins for ins in pm.instruments if ins.is_drum and ins.notes]
    signals = {"n_melodic_tracks": len(melodic), "n_drum_tracks": len(drums)}

    if not melodic:
        if drums:
            return "drums", 1.0, [], signals
        return "piano", 0.3, ["empty"], signals

    n_bends = sum(len(ins.pitch_bends) for ins in melodic)
    signals["n_pitch_bends"] = n_bends
    cc2 = max((_cc_density(ins, 2) for ins in melodic), default=0.0)
    signals["cc2_density"] = round(cc2, 2)

    if len(melodic) >= 3:
        # multitrack with bend -> guitar/MPE family; without bend still merged as multitrack
        conf = 0.9 if n_bends > 0 else 0.6
        return "multichannel", conf, (["piano"] if conf < 0.7 else []), signals

    ks, _base = _find_keyswitch_cluster(melodic[0].notes) if len(melodic) == 1 else (set(), None)
    if ks:
        signals["keyswitch_pitches"] = sorted(ks)
        return "keyswitch", 0.85, [], signals

    if cc2 >= 5.0:  # ≥5 CC2 events per second = continuous breath curve
        return "breath", 0.8, ["mono_lead"], signals

    if n_bends >= 8:
        return "mono_lead", 0.8, ["piano"], signals

    return "piano", 0.9, [], signals


# ---- bend curve -> art label ----


def _classify_bend(ins, note):
    """bend curve within a note span -> art label (0/1/2). Check slide first (large and monotonic net displacement), then vibrato."""
    pts = [
        (b.time, b.pitch / 8192.0 * BEND_RANGE_ST)
        for b in ins.pitch_bends
        if note.start <= b.time < note.end
    ]
    if len(pts) < 3:
        return 0
    vals = np.array([v for _, v in pts])
    dur = max(note.end - note.start, 1e-6)

    steps = np.diff(vals)
    steps = steps[np.abs(steps) > 1e-9]
    net = vals[-1] - vals[0]
    if len(steps) and abs(net) >= SLIDE_MIN_ST:
        mono = np.mean(np.sign(steps) == np.sign(net))
        if mono >= SLIDE_MONO_FRAC:
            return 2  # slide

    centered = vals - vals.mean()
    if np.max(np.abs(centered)) >= VIB_MIN_AMP_ST:
        crossings = np.sum(np.diff(np.sign(centered)) != 0)
        rate = crossings / 2.0 / dur
        if VIB_RATE_LO <= rate <= VIB_RATE_HI:
            return 1  # vibrato
    return 0


# ---- Monophonisation ----


def _monophonize(events, report):
    """Sort, remove chords (keep the highest note), truncate overlaps. events: [(onset, dur, midi, art, extra)]"""
    events = sorted(events, key=lambda e: (e[0], -e[2]))
    out = []
    for ev in events:
        if out and abs(ev[0] - out[-1][0]) <= CHORD_EPS:
            report.n_chord_notes_dropped += 1  # same onset: highest note already kept
            continue
        out.append(list(ev))
    for i in range(len(out) - 1):
        end_i = out[i][0] + out[i][1]
        if end_i > out[i + 1][0]:
            out[i][1] = max(0.02, out[i + 1][0] - out[i][0])
            report.n_overlaps_trimmed += 1
    return out


# ---- Per-note CC energy hints ----


def _cc_energy_per_note(ins, notes_events, cc_number):
    ccs = sorted([(c.time, c.value) for c in ins.control_changes if c.number == cc_number])
    if not ccs:
        return None
    t = np.array([x for x, _ in ccs])
    v = np.array([y for _, y in ccs], dtype=float) / 127.0
    vals = []
    for onset, dur, *_ in notes_events:
        m = (t >= onset) & (t < onset + dur)
        vals.append(
            float(v[m].mean())
            if m.any()
            else float(v[np.searchsorted(t, onset).clip(0, len(v) - 1)])
        )
    return vals


# ---- Main entry point ----


def import_midi(path, ks_map=None, bend_range_st=None):
    """Parse a MIDI file of any dialect -> (score, ImportReport).

    score can be fed straight to neural.perform.control_from_behavior (extra keys are ignored by the engine).
    Raises DrumFileError for pure drum files.
    """
    global BEND_RANGE_ST
    if bend_range_st is not None:
        BEND_RANGE_ST = float(bend_range_st)
    ks_map = dict(DEFAULT_KS_MAP if ks_map is None else ks_map)

    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(path))
    report = ImportReport()
    dialect, conf, cands, signals = detect_dialect(pm)
    report.dialect, report.confidence, report.candidates = dialect, conf, cands

    if dialect == "drums":
        raise DrumFileError(
            "Pure drum file: note numbers are instrument identities (36=kick/38=snare), not pitches; "
            "Quill's pitched monophonic pipeline does not apply."
        )

    drums = [ins for ins in pm.instruments if ins.is_drum and ins.notes]
    if drums:
        report.n_drum_tracks_dropped = len(drums)
        report.warnings.append(f"Dropped {len(drums)} drum tracks")

    melodic = _melodic_instruments(pm)
    report.n_notes_in = sum(len(ins.notes) for ins in melodic)

    events = []  # (onset, dur, midi, art, cc_energy_or_None)

    if dialect == "keyswitch":
        ins = melodic[0]
        ks_pitches, base = _find_keyswitch_cluster(ins.notes)
        switches = sorted([(n.start, n.pitch) for n in ins.notes if n.pitch in ks_pitches])
        report.n_keyswitches_stripped = len(switches)
        report.ks_map_used = {
            int(p): ART_NAMES.get(ks_map.get(p - base, 0), "plain") for p in ks_pitches
        }
        unknown = {p - base for _, p in switches} - set(ks_map)
        if unknown:
            report.warnings.append(f"Unmapped keyswitch offsets {sorted(unknown)} -> treated as plain")
        sw_t = [t for t, _ in switches]
        sw_art = [ks_map.get(p - base, 0) for _, p in switches]
        for n in ins.notes:
            if n.pitch in ks_pitches:
                continue
            k = int(np.searchsorted(sw_t, n.start, side="right")) - 1
            art = sw_art[k] if k >= 0 else 0  # latching: default plain before the first KS
            events.append((n.start, n.end - n.start, n.pitch, art, None))

    elif dialect in ("multichannel", "mono_lead"):
        report.n_channels_merged = len(melodic) if dialect == "multichannel" else 0
        for ins in melodic:
            for n in ins.notes:
                art = _classify_bend(ins, n)
                if art == 2:
                    report.n_bend_slide += 1
                elif art == 1:
                    report.n_bend_vibrato += 1
                events.append((n.start, n.end - n.start, n.pitch, art, None))

    else:  # piano / breath
        ins = melodic[0]
        for n in ins.notes:
            events.append((n.start, n.end - n.start, n.pitch, 0, None))

    # per-note CC energy hints: for single-instrument dialects take the densest continuous CC curve,
    # wind-style prefers CC2 (breath), sound-module style prefers CC11 (expression)/CC1 (mod).
    if len(melodic) == 1 and dialect != "multichannel":
        ins = melodic[0]
        for cc_num, name in ((2, "cc2"), (11, "cc11"), (1, "cc1")):
            if _cc_density(ins, cc_num) >= 5.0:
                raw = [(e[0], e[1]) for e in events]
                cc_vals = _cc_energy_per_note(ins, raw, cc_num)
                if cc_vals:
                    report.cc_energy_source = name
                    events = [(o, d, m, a, cc_vals[i]) for i, (o, d, m, a, _) in enumerate(events)]
                break

    events = _monophonize(events, report)
    score = []
    for onset, dur, midi, art, cc in events:
        note = {
            "midi": int(midi),
            "dur": round(float(dur), 4),
            "art": int(art),
            "onset": round(float(onset), 4),
        }
        if cc is not None:
            note["cc_energy"] = round(float(cc), 3)
        score.append(note)
    report.n_notes_out = len(score)
    return score, report
