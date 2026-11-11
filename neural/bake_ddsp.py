"""Desktop "bake": one capture recording -> a DDSP timbre fine-tuned from the pretrained base (subprocess, MPS).

L0 capture takes seconds and is real-time, so it is the default path; DDSP can learn textures such as friction noise and amplitude fluctuation
that a single steady spectrum cannot capture, but it needs minutes of training, so it is an optional background job after capture. Desktop only,
not web (the web render container has no GPU).

  python -m neural.bake_ddsp --wav take.wav --name myglass --base glass --steps 1500 \
      --out assets/user_ddsp --status assets/user_ddsp/_bake_status.json

Produces <out>/<name>.pt + <name>.json (base/envelope/steps/val). NeuralTake loads it as "ddsp:<name>";
the behaviour library defaults to the base's trajectory library (or a user_styles/<name>.npz of the same name).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

QUILL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QUILL))

from neural.ddsp import HOP, SR, extract_features  # noqa: E402
from neural.neural_take import INSTRUMENTS  # noqa: E402
from neural.train_ddsp import _trim_silence, train  # noqa: E402

MIN_TAKE_S = 8.0  # any shorter and there is not enough data for a validation split; below 20s the UI warns about possible overfitting


def chunk_dataset(y, sr_in, chunk_s=3.0, hop_s=1.5, min_s=0.9):
    """Whole recording -> overlapping slices treated as "notes" (train() random-crops per note for augmentation + validation split).

    Features (PESTO f0 / log-RMS) are extracted once for the whole recording and then sliced by frame, an order of magnitude faster than per slice;
    loudness standardisation is therefore computed over the whole recording, consistent with training.
    """
    y = _trim_silence(np.asarray(y, np.float32), sr_in)
    f0, loud, au = extract_features(y, sr_in)
    fps = SR / HOP
    T = f0.shape[1]
    n, h, Tmin = int(chunk_s * fps), int(hop_s * fps), int(min_s * fps)
    out = []
    for s in range(0, max(1, T - n // 2), h):
        e = min(T, s + n)
        if e - s < Tmin:
            continue
        out.append((f0[:, s:e], loud[:, s:e], au[:, s * HOP : e * HOP]))
    return out


def bake(wav, name, base="violin", steps=1500, out_dir="assets/user_ddsp", status=None, lr=3e-4):
    from analysis.io import load

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status_p = Path(status) if status else None
    t0 = time.time()

    def write(state, **kw):
        if status_p is None:
            return
        d = {"state": state, "name": name, "base": base, "steps": steps, "t": round(time.time() - t0, 1)}
        d.update(kw)
        tmp = status_p.with_suffix(".tmp")
        tmp.write_text(json.dumps(d))
        tmp.replace(status_p)  # atomic write, the UI polling never reads a partial file

    try:
        write("preparing", step=0)
        # a capture take itself is ≤60 s; allow more when the CLI is fed a long file
        y, sr = load(str(wav), sr=44100, max_seconds=120.0)
        take_s = len(y) / sr
        if take_s < MIN_TAKE_S:
            write("failed", msg=f"take {take_s:.1f}s < {MIN_TAKE_S:.0f}s")
            return 1
        # PESTO pitch tracking runs at about 1× real time (CPU) and dominates the preparing stage; the UI sets expectations from this
        write("preparing", step=0, take_s=round(take_s, 1), eta_s=round(take_s * 1.1))
        data = chunk_dataset(y, sr)
        if len(data) < 2:
            write("failed", msg="not enough voiced material")
            return 1
        write("training", step=0, chunks=len(data), take_s=round(take_s, 1))
        spec = INSTRUMENTS[base]
        init = torch.load(str(spec["ckpt"]), map_location="cpu", weights_only=True)
        partial = out_dir / f"{name}.partial.pt"

        def on_step(it, loss, best):
            bv = None if best == float("inf") else round(best, 2)
            write("training", step=it, loss=round(loss, 2), best_val=bv)

        model, losses, best_val, _ = train(
            data,
            steps=steps,
            batch=6,
            lr=lr,
            crop_s=1.0,
            ckpt_path=str(partial),
            init_state=init,
            verbose=False,
            on_step=on_step,
        )
        state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(state, out_dir / f"{name}.pt")
        bv = None if best_val == float("inf") else round(float(best_val), 3)
        (out_dir / f"{name}.json").write_text(
            json.dumps(
                {
                    "name": name,
                    "base": base,
                    "envelope": spec["envelope"],
                    "steps": steps,
                    "best_val": bv,
                    "train_loss": round(float(np.mean(losses[-100:])), 3),
                    "src": str(wav),
                    "take_s": round(take_s, 1),
                    "chunks": len(data),
                    "train_s": round(time.time() - t0, 1),
                }
            )
        )
        if partial.exists():
            partial.unlink()
        write("done", step=steps, best_val=bv)
        return 0
    except Exception as e:
        # failures in the subprocess must land in the status file so the UI can see them
        write("failed", msg=str(e)[:120])
        return 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", default="violin", choices=sorted(INSTRUMENTS))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--out", default=str(QUILL / "assets" / "user_ddsp"))
    ap.add_argument("--status", default=None)
    a = ap.parse_args(argv)
    return bake(a.wav, a.name, a.base, a.steps, a.out, a.status)


if __name__ == "__main__":
    sys.exit(main())
