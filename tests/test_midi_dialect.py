"""MIDI dialect adapter tests: synthetic files for six dialects -> correct detection + conversion."""

import sys
from pathlib import Path

import numpy as np
import pytest

pretty_midi = pytest.importorskip("pretty_midi")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.midi_dialect import (  # noqa: E402
    DrumFileError,
    import_midi,
)


def _note(pitch, start, end, vel=90):
    return pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=end)


def _save(pm, tmp_path, name):
    p = tmp_path / name
    pm.write(str(p))
    return p


# ---- piano style: pass-through ---- #


def test_piano_passthrough(tmp_path):
    pm = pretty_midi.PrettyMIDI()
    ins = pretty_midi.Instrument(program=0)
    for i, pitch in enumerate([60, 62, 64, 65, 67]):
        ins.notes.append(_note(pitch, i * 0.5, i * 0.5 + 0.45))
    pm.instruments.append(ins)
    score, rep = import_midi(_save(pm, tmp_path, "piano.mid"))
    assert rep.dialect == "piano"
    assert rep.n_notes_out == 5
    assert [n["midi"] for n in score] == [60, 62, 64, 65, 67]
    assert all(n["art"] == 0 for n in score)
    assert score[1]["onset"] == pytest.approx(0.5, abs=1e-3)


# ---- keyswitch style: stripping + latching mapping ---- #


def test_keyswitch_strip_and_latch(tmp_path):
    pm = pretty_midi.PrettyMIDI()
    ins = pretty_midi.Instrument(program=40)  # violin
    # keyswitch cluster: C1=24(plain) C#1=25(vibrato) D1=26(slide), short
    ins.notes.append(_note(25, 0.00, 0.05))  # -> vibrato takes effect
    ins.notes.append(_note(60, 0.10, 0.60))
    ins.notes.append(_note(62, 0.70, 1.20))
    ins.notes.append(_note(24, 1.25, 1.30))  # -> plain takes effect
    ins.notes.append(_note(64, 1.40, 1.90))
    ins.notes.append(_note(26, 1.95, 2.00))  # -> slide takes effect
    ins.notes.append(_note(65, 2.10, 2.60))
    pm.instruments.append(ins)
    score, rep = import_midi(_save(pm, tmp_path, "ks.mid"))
    assert rep.dialect == "keyswitch"
    assert rep.n_keyswitches_stripped == 3
    assert rep.n_notes_out == 4  # all fake notes stripped
    assert [n["midi"] for n in score] == [60, 62, 64, 65]
    assert [n["art"] for n in score] == [1, 1, 0, 2]  # latching


# ---- guitar/multichannel style: merge + bend -> slide ---- #


def test_multichannel_merge_and_bend_slide(tmp_path):
    pm = pretty_midi.PrettyMIDI()
    onset = 0.0
    for s in range(3):  # three "strings"
        ins = pretty_midi.Instrument(program=25)
        for k in range(2):
            ins.notes.append(_note(52 + s * 5 + k, onset, onset + 0.4))
            onset += 0.5
        pm.instruments.append(ins)
    # the last note of the last string gets a monotonic bend of +2 semitones -> slide
    tgt = pm.instruments[-1]
    n = tgt.notes[-1]
    for i, t in enumerate(np.linspace(n.start, n.end - 0.01, 12)):
        tgt.pitch_bends.append(pretty_midi.PitchBend(pitch=int(8191 * i / 11), time=float(t)))
    score, rep = import_midi(_save(pm, tmp_path, "gtr.mid"))
    assert rep.dialect == "multichannel"
    assert rep.n_channels_merged == 3
    assert rep.n_notes_out == 6
    assert rep.n_bend_slide == 1
    assert score[-1]["art"] == 2
    assert [n["onset"] for n in score] == sorted(n["onset"] for n in score)


# ---- monophonic lead: periodic bend -> vibrato ---- #


def test_mono_lead_bend_vibrato(tmp_path):
    pm = pretty_midi.PrettyMIDI()
    ins = pretty_midi.Instrument(program=80)
    ins.notes.append(_note(69, 0.0, 2.0))
    t = np.linspace(0.0, 1.99, 80)
    for ti, v in zip(t, 0.25 * np.sin(2 * np.pi * 6.0 * t)):  # 6Hz ±0.25st
        ins.pitch_bends.append(pretty_midi.PitchBend(pitch=int(v / 2.0 * 8192), time=float(ti)))
    pm.instruments.append(ins)
    score, rep = import_midi(_save(pm, tmp_path, "lead.mid"))
    assert rep.dialect == "mono_lead"
    assert rep.n_bend_vibrato == 1
    assert score[0]["art"] == 1


# ---- wind style: CC2 -> per-note energy hints ---- #


def test_breath_cc2_energy(tmp_path):
    pm = pretty_midi.PrettyMIDI()
    ins = pretty_midi.Instrument(program=66)
    ins.notes.append(_note(70, 0.0, 1.0))
    ins.notes.append(_note(72, 1.1, 2.1))
    for t in np.arange(0.0, 2.2, 0.05):  # continuous breath curve at 20/s
        val = 40 if t < 1.05 else 110
        ins.control_changes.append(pretty_midi.ControlChange(number=2, value=val, time=float(t)))
    pm.instruments.append(ins)
    score, rep = import_midi(_save(pm, tmp_path, "ewi.mid"))
    assert rep.dialect == "breath"
    assert rep.cc_energy_source == "cc2"
    assert score[0]["cc_energy"] < score[1]["cc_energy"]


# ---- drums: explicitly rejected ---- #


def test_drums_rejected(tmp_path):
    pm = pretty_midi.PrettyMIDI()
    ins = pretty_midi.Instrument(program=0, is_drum=True)
    for i in range(4):
        ins.notes.append(_note(36 if i % 2 == 0 else 38, i * 0.25, i * 0.25 + 0.1))
    pm.instruments.append(ins)
    with pytest.raises(DrumFileError):
        import_midi(_save(pm, tmp_path, "drums.mid"))


# ---- mixed file: drum track dropped, melody track kept ---- #


def test_mixed_drops_drum_track(tmp_path):
    pm = pretty_midi.PrettyMIDI()
    mel = pretty_midi.Instrument(program=0)
    for i, p in enumerate([60, 64, 67]):
        mel.notes.append(_note(p, i * 0.5, i * 0.5 + 0.4))
    drum = pretty_midi.Instrument(program=0, is_drum=True)
    drum.notes.append(_note(36, 0.0, 0.1))
    pm.instruments.extend([mel, drum])
    score, rep = import_midi(_save(pm, tmp_path, "mixed.mid"))
    assert rep.n_drum_tracks_dropped == 1
    assert rep.n_notes_out == 3


# ---- monophonisation: chords keep the highest note, overlaps truncated ---- #


def test_monophonize_chord_and_overlap(tmp_path):
    pm = pretty_midi.PrettyMIDI()
    ins = pretty_midi.Instrument(program=0)
    ins.notes.append(_note(60, 0.0, 1.0))  # chord: same onset as 64
    ins.notes.append(_note(64, 0.0, 1.0))  # -> keep 64
    ins.notes.append(_note(67, 0.6, 1.4))  # overlaps the previous note -> previous note truncated
    pm.instruments.append(ins)
    score, rep = import_midi(_save(pm, tmp_path, "poly.mid"))
    assert rep.n_chord_notes_dropped == 1
    assert rep.n_overlaps_trimmed == 1
    assert [n["midi"] for n in score] == [64, 67]
    assert score[0]["dur"] == pytest.approx(0.6, abs=1e-3)


# ---- score compatible with the engine IR: correct key set ---- #


def test_score_ir_contract(tmp_path):
    pm = pretty_midi.PrettyMIDI()
    ins = pretty_midi.Instrument(program=0)
    ins.notes.append(_note(60, 0.0, 0.5))
    pm.instruments.append(ins)
    score, _ = import_midi(_save(pm, tmp_path, "ir.mid"))
    n = score[0]
    assert set(n) >= {"midi", "dur", "art", "onset"}
    assert isinstance(n["midi"], int) and isinstance(n["art"], int)
    assert n["art"] in (0, 1, 2)


# ---- sound-module style: a CC11 curve also yields energy hints (not limited to CC2) ---- #


def test_cc11_energy_on_keyswitch_dialect(tmp_path):
    pm = pretty_midi.PrettyMIDI()
    ins = pretty_midi.Instrument(program=48)
    ins.notes.append(_note(24, 0.0, 0.05))  # keyswitch → plain
    ins.notes.append(_note(60, 0.1, 1.0))
    ins.notes.append(_note(62, 1.1, 2.0))
    ins.notes.append(_note(64, 2.05, 2.08))  # reach the detection threshold (≥4 notes)
    for t in np.arange(0.0, 2.1, 0.05):
        val = 30 if t < 1.05 else 100
        ins.control_changes.append(pretty_midi.ControlChange(number=11, value=val, time=float(t)))
    pm.instruments.append(ins)
    score, rep = import_midi(_save(pm, tmp_path, "cc11.mid"))
    assert rep.dialect == "keyswitch"
    assert rep.cc_energy_source == "cc11"
    assert score[0]["cc_energy"] < score[1]["cc_energy"]
