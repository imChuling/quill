"""IDMT-SMT-Guitar formal evaluation: FR-6 dual criteria (stage 8, brought forward).

Three-way ablation: rule tree vs random forest vs MERT probe, macro F1 + confusion matrix on the public IDMT set.
Exit criteria (product plan success metrics): rule tree macro F1≥0.7 / MERT probe≥0.8.

The dataset must be downloaded manually (Fraunhofer IDMT-SMT-Guitar, several GB); a friendly message is shown when missing.
Usage:
  python -m eval.eval_idmt --root /path/to/IDMT-SMT-GUITAR_V2 [--mert] [--max-files 40]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.articulation import classify_events, forest_predict, train_forest  # noqa: E402
from analysis.datasets import available, load_idmt_smt_guitar  # noqa: E402
from analysis.features import compute_event_features  # noqa: E402
from eval.metrics import classification_report, plot_confusion  # noqa: E402

SR = 44100
DEMO_DIR = Path(__file__).resolve().parents[1] / "assets" / "demos"


def _find_wav(root, stem):
    """Find the audio file with the same name as the annotation in the dataset tree (wav preferred)."""
    for ext in (".wav", ".WAV", ".flac", ".mp3"):
        hits = glob.glob(os.path.join(root, "**", stem + ext), recursive=True)
        if hits:
            return hits[0]
    return None


def _load_events_with_features(root, max_files=None, want_audio_for_mert=False):
    from analysis.io import load

    events_all = load_idmt_smt_guitar(root)
    if not events_all:
        return [], {}
    by_file = defaultdict(list)
    for e in events_all:
        by_file[e.get("_source_file", "?")].append(e)

    files = list(by_file)
    if max_files and max_files < len(files):
        import random

        random.Random(0).shuffle(files)  # shuffle so a subset still covers slide/sustain
        files = files[:max_files]
    out, audio_by_file = [], {}
    n = len(files)
    for k, f in enumerate(files):
        if k % 25 == 0:
            print(f"  feature extraction {k}/{n} files, {len(out)} events so far ...", flush=True)
        stem = os.path.splitext(f)[0]
        wav = _find_wav(root, stem)
        if wav is None:
            continue
        try:
            y, _ = load(wav, sr=SR, do_normalize=True, max_seconds=60)
        except Exception:
            continue
        evs = by_file[f]
        compute_event_features(y, SR, evs)  # add features + f0_track
        out.extend(evs)
        if want_audio_for_mert:
            audio_by_file[f] = y
    return out, audio_by_file


def main(argv=None):
    ap = argparse.ArgumentParser(description="IDMT formal evaluation (rules vs forest vs MERT)")
    ap.add_argument("--root", required=True, help="IDMT-SMT-Guitar root directory")
    ap.add_argument("--mert", action="store_true", help="add the MERT probe (slow, embeddings extracted per event)")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--refresh", action="store_true", help="ignore the cache and re-extract features")
    args = ap.parse_args(argv)

    if not available(args.root):
        print(f"[missing] IDMT dataset not found: {args.root}")
        print("  Download Fraunhofer IDMT-SMT-Guitar and pass --root pointing to the extracted directory.")
        return 2

    # feature cache (extraction takes ~20min, reruns take seconds once cached; --refresh forces a refresh)
    import hashlib
    import pickle
    from collections import Counter, defaultdict

    key = hashlib.md5(f"{args.root}|{args.max_files}".encode()).hexdigest()[:12]
    cache = Path(f"/tmp/quill_idmt_cache_{key}.pkl")
    audio_by_file = {}
    if cache.exists() and not args.refresh:
        print(f"Loading feature cache {cache.name} ...")
        with open(cache, "rb") as fh:
            events = pickle.load(fh)
    else:
        print("Loading IDMT annotations + per-event features ...")
        events, audio_by_file = _load_events_with_features(
            args.root, args.max_files, want_audio_for_mert=args.mert
        )
        if events:
            with open(cache, "wb") as fh:
                pickle.dump(events, fh)
    if not events:
        print("[empty] parsed 0 events with audio; check the directory layout / whether audio file names match the annotations.")
        return 2
    true = [e["label"] for e in events]
    print(f"  {len(events)} events, distribution: {dict(Counter(true))}")

    results = {}

    # rule tree: **classify grouped by file** (avoids cross-file t_on collisions being misjudged as strum: event onsets in many files
    # are concentrated at 0.2-2.5s, and mixing them all together would trip the same-window strum rule).
    present = set(true)  # label space actually present in the dataset
    by_file = defaultdict(list)
    for e in events:
        by_file[e.get("_source_file", "?")].append(e)
    for evs in by_file.values():
        classify_events(evs, valid_labels=present)
    rule_pred = [e["label"] for e in events]
    results["rule"] = classification_report(true, rule_pred)
    plot_confusion(true, rule_pred, DEMO_DIR / "idmt_confusion_rule.png", title="IDMT rule-tree")

    # random forest + MERT: stratified train/test
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(events))
    tr, te = train_test_split(idx, test_size=0.35, random_state=0, stratify=true)
    ytr = [true[i] for i in tr]
    yte = [true[i] for i in te]

    etr = [events[i] for i in tr]
    ete = [events[i] for i in te]
    clf = train_forest(etr, ytr)
    pf = forest_predict(clf, ete)
    results["forest"] = classification_report(yte, pf)
    plot_confusion(yte, pf, DEMO_DIR / "idmt_confusion_forest.png", title="IDMT forest")

    if args.mert:
        print("Extracting MERT embeddings (per event, slow) ...")
        from analysis.io import load as _load
        from analysis.mert_probe import mert_embed, probe_predict, train_probe

        # embeddings are extracted per event from its file's audio; on a cache hit audio_by_file is empty -> reread the wav on demand
        audio_cache = dict(audio_by_file)
        emb = []
        for k, e in enumerate(events):
            if k % 200 == 0:
                print(f"  MERT embeddings {k}/{len(events)} ...", flush=True)
            sf_ = e.get("_source_file")
            y = audio_cache.get(sf_)
            if y is None:
                wav = _find_wav(args.root, os.path.splitext(sf_)[0]) if sf_ else None
                if wav:
                    try:
                        y, _ = _load(wav, sr=SR, max_seconds=60)
                    except Exception:
                        y = None
                audio_cache[sf_] = y
            if y is None:
                emb.append(np.zeros(768, np.float32))
                continue
            a, b = int(e["t_on"] * SR), int(e["t_off"] * SR)
            seg = y[a:b] if b > a else y[a : a + 1]
            emb.append(mert_embed(seg, SR))
        emb = np.array(emb)
        pipe = train_probe(emb[tr], ytr)
        pm = probe_predict(pipe, emb[te])
        results["mert"] = classification_report(yte, pm)
        plot_confusion(yte, pm, DEMO_DIR / "idmt_confusion_mert.png", title="IDMT MERT probe")

    # summary
    print("\n---------------- IDMT three-way ablation (macro F1) ----------------")
    targets = {"rule": 0.7, "mert": 0.8}
    for k in ("rule", "forest", "mert"):
        if k not in results:
            continue
        r = results[k]
        tgt = targets.get(k)
        flag = "" if tgt is None else (" PASS" if r["macro_f1"] >= tgt else " <target")
        print(
            f"  {k:8s}: macro F1={r['macro_f1']:.3f}  acc={r['accuracy']:.3f}"
            + (f"  (target≥{tgt}{flag})" if tgt else "")
        )
    print(f"Confusion matrix plots -> {DEMO_DIR}/idmt_confusion_*.png")
    print("Criteria (product plan): rule tree≥0.7 / MERT probe≥0.8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
