"""Hand-labelling tool + recall computation: FR-5 acceptance (stage 4 Day4).

* compute_recall(detected, reference): recall/precision by onset + pitch matching (automatic acceptance & comparison with hand labels)
* LabelTool: matplotlib click-to-play + key labelling (label 50 events for the confusion matrix/recall, for real material)

Keys: p=pluck s=slide l=legato t=strum u=sustain  arrow keys switch event  Enter=play  w=save
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Recall / precision (onset + pitch matching, greedy)
# --------------------------------------------------------------------------- #
def compute_recall(detected, reference, onset_tol_s=0.05, pitch_tol=1.0):
    """Match each reference event to one detected event (onset ±tol, pitch ±tol).

    reference / detected: [{"t_on":..,"pitch_midi":..}, ...]
    Returns {"recall","precision","n_ref","n_det","matched", "misses":[...]}.
    """
    det = sorted(detected, key=lambda e: e["t_on"])
    used = [False] * len(det)
    matched = 0
    misses = []
    for r in sorted(reference, key=lambda e: e["t_on"]):
        rp = r.get("pitch_midi")
        best, best_dt = None, onset_tol_s + 1e-9
        for j, d in enumerate(det):
            if used[j]:
                continue
            dt = abs(d["t_on"] - r["t_on"])
            if dt > onset_tol_s:
                continue
            dp = d.get("pitch_midi")
            if rp is not None and dp is not None and abs(dp - rp) > pitch_tol:
                continue
            if dt < best_dt:
                best, best_dt = j, dt
        if best is not None:
            used[best] = True
            matched += 1
        else:
            misses.append(r)
    n_ref, n_det = len(reference), len(detected)
    return {
        "recall": matched / n_ref if n_ref else 0.0,
        "precision": matched / n_det if n_det else 0.0,
        "n_ref": n_ref,
        "n_det": n_det,
        "matched": matched,
        "misses": misses,
    }


# --------------------------------------------------------------------------- #
# Interactive hand-labelling tool (for real material; unavailable without a display)
# --------------------------------------------------------------------------- #
_LABEL_KEYS = {"p": "pluck", "s": "slide", "l": "legato", "t": "strum", "u": "sustain"}


class LabelTool:
    def __init__(self, y, sr, events, out_path="assets/demos/labels.json"):
        self.y = np.asarray(y, dtype=np.float32)
        self.sr = int(sr)
        self.events = events
        self.out_path = Path(out_path)
        self.idx = 0
        self.labels = {e["id"]: e.get("label", "") for e in events}

    def run(self):  # pragma: no cover - interactive, not run in CI
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 4))
        t = np.arange(len(self.y)) / self.sr
        ax.plot(t, self.y, lw=0.4, color="gray")
        for e in self.events:
            ax.axvline(e["t_on"], color="C0", lw=0.8, alpha=0.6)
        self._cursor = ax.axvline(self.events[0]["t_on"], color="r", lw=1.5)
        self.ax, self.fig = ax, fig
        self._refresh_title()
        fig.canvas.mpl_connect("key_press_event", self._on_key)
        plt.show()

    def _refresh_title(self):  # pragma: no cover
        e = self.events[self.idx]
        self.ax.set_title(
            f"event {self.idx + 1}/{len(self.events)}  t={e['t_on']:.2f}s "
            f"pitch={e.get('pitch_midi')}  label={self.labels[e['id']] or '?'}  "
            f"[p/s/l/t/u label, ←→ switch, Enter play, w save]"
        )
        self.fig.canvas.draw_idle()

    def _play(self):  # pragma: no cover
        import sounddevice as sd

        e = self.events[self.idx]
        a, b = int(e["t_on"] * self.sr), int(e["t_off"] * self.sr)
        sd.play(self.y[a:b], self.sr)

    def _on_key(self, ev):  # pragma: no cover
        if ev.key in _LABEL_KEYS:
            self.labels[self.events[self.idx]["id"]] = _LABEL_KEYS[ev.key]
        elif ev.key in ("right", "n"):
            self.idx = min(self.idx + 1, len(self.events) - 1)
        elif ev.key in ("left", "b"):
            self.idx = max(self.idx - 1, 0)
        elif ev.key == "enter":
            self._play()
            return
        elif ev.key == "w":
            self.save()
            print(f"saved -> {self.out_path}")
            return
        self._cursor.set_xdata([self.events[self.idx]["t_on"]] * 2)
        self._refresh_title()

    def save(self):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_path, "w", encoding="utf-8") as fh:
            json.dump(self.labels, fh, ensure_ascii=False, indent=2)
        return self.out_path
