"""SnapshotLibrary: labelled TimbreSnapshot corpus and preset library.

Two uses: retrieve one representative snapshot by tag (bass / guitar / violin...) or semantic descriptor and play it directly;
or flatten the (tag, instrument, pitch) metadata and all snapshots into a parameter matrix for training a conditional generative prior,
while the synthesis side remains deterministic real-time DSP.

The corpus is a directory of .npz snapshots plus index.json (tag/family/instrument/midi/source metadata).
Data sources are tinysol_groups / nsynth_groups from eval.fidelity.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analysis.timbre import analyze_timbre
from snapshot import load_snapshot, save_snapshot

SR = 44100


def _slug(s):
    return s.replace("/", "_").replace(" ", "")


def build_corpus(groups, out_dir, source="dataset", quasi=False, verbose=True):
    """groups: [(name, [(midi, audio), ...]), ...], name like 'TinySOL/Strings/Violin'.

    Per sample analyze_timbre -> save .npz -> accumulate index.json. Returns index (list of dict).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx_path = out_dir / "index.json"
    index = json.loads(idx_path.read_text()) if idx_path.exists() else []
    seen = {(e["instrument"], e["midi"]) for e in index}

    for name, samples in groups:
        parts = name.split("/")
        tag = parts[-1]
        family = parts[1] if len(parts) > 2 else (parts[0] if parts else name)
        for midi, audio in samples:
            if (name, int(midi)) in seen:
                continue
            f0_hint = 440.0 * 2 ** ((int(midi) - 69) / 12.0)  # ground-truth pitch fixes octave errors
            try:
                snap = analyze_timbre(
                    np.asarray(audio, np.float32),
                    sr=SR,
                    name=f"{name}@{midi}",
                    quasi_harmonic=quasi,
                    f0_hint=f0_hint,
                )
            except Exception as e:
                if verbose:
                    print(f"  [skip] {name}@{midi}: {e}")
                continue
            if not (np.isfinite(snap["f0_ref"]) and snap["f0_ref"] > 0):
                continue
            fn = f"{_slug(name)}__{int(midi)}.npz"
            save_snapshot(out_dir / fn, snap)
            entry = {
                "file": fn,
                "tag": tag,
                "family": family,
                "instrument": name,
                "midi": int(midi),
                "f0_ref": float(snap["f0_ref"]),
                "source": source,
            }
            index.append(entry)
            seen.add((name, int(midi)))
            if verbose:
                print(f"  + {fn}  f0={snap['f0_ref']:.1f}Hz")
    idx_path.write_text(json.dumps(index, ensure_ascii=False, indent=0))
    if verbose:
        print(f"corpus {len(index)} snapshots -> {idx_path}")
    return index


# quality whitelist: families that resynthesise well with spectral modelling: strong resonance, sustained tones, formant preservation helps. Excluded because they collapse:
# mallet inharmonic percussion, accordion free reeds, bowed-string vibrato plus bow noise, breathy flute/double reeds. A plucked guitar's identity lies in
# the pluck transient, which a static snapshot cannot capture, so it is not whitelisted and guitar goes through the DDSP neural timbre; low-register plucked bass is acceptable and kept.
GOOD_TAGS = {
    "bass",
    "brass",
    "organ",
    "keyboard",
    "vocal",
    "reed",
    "Bass_Tuba",
    "Contrabass",
    "Bassoon",
    "Horn",
    "Trombone",
    "Trumpet_C",
    "Clarinet_Bb",
    "Sax_Alto",
}

# semantic descriptor -> candidate tags: choose timbres by perceived sound rather than instrument names, all within GOOD_TAGS
DESCRIPTORS = {
    "low": ["bass", "Contrabass", "Bass_Tuba"],
    "brass": ["brass", "Trumpet_C", "Trombone", "Horn", "Bass_Tuba"],
    "warm": ["Bassoon", "Horn", "Contrabass", "organ"],
    "bright": ["Trumpet_C", "reed", "Sax_Alto"],
    "reedy": ["reed", "Clarinet_Bb", "Sax_Alto"],
    "vocal": ["vocal"],
    "organ": ["organ", "keyboard"],
}


class SnapshotLibrary:
    """Load the corpus; retrieve playable representative snapshots by tag or semantic descriptor."""

    def __init__(self, root):
        self.root = Path(root)
        self.index = json.loads((self.root / "index.json").read_text())

    def tags(self):
        return sorted({e["tag"] for e in self.index})

    def entries(self, tag=None, family=None):
        out = self.index
        if tag is not None:
            out = [e for e in out if e["tag"] == tag]
        if family is not None:
            out = [e for e in out if e["family"] == family]
        return out

    def retrieve(self, tag=None, family=None, pitch_midi=60):
        """Take the representative snapshot closest to the target pitch; formant preservation lets it play across the whole range."""
        cand = self.entries(tag=tag, family=family)
        if not cand:
            raise KeyError(f"no snapshot for tag={tag} family={family}")
        f0_tgt = 440.0 * 2 ** ((pitch_midi - 69) / 12)
        best = min(cand, key=lambda e: abs(np.log2(max(e["f0_ref"], 1.0) / f0_tgt)))
        return load_snapshot(self.root / best["file"]), best

    def good_entries(self):
        """Entries in the quality whitelist; demos and retrieval expose only these by default."""
        return [e for e in self.index if e["tag"] in GOOD_TAGS]

    def descriptors(self):
        """Available semantic descriptors (with ≥1 matching good snapshot in the corpus) -> {descriptor: count}."""
        out = {}
        for desc, tags in DESCRIPTORS.items():
            n = sum(1 for e in self.good_entries() if e["tag"] in tags)
            if n:
                out[desc] = n
        return out

    def browse(self):
        """Browse the library: returns [(descriptor, count, representative families)]."""
        rows = []
        for desc, n in self.descriptors().items():
            tags = DESCRIPTORS[desc]
            fams = sorted({e["tag"] for e in self.good_entries() if e["tag"] in tags})
            rows.append((desc, n, fams))
        return rows

    def retrieve_descriptor(self, descriptor, pitch_midi=60):
        """Retrieve the good snapshot closest in pitch by semantic descriptor ("warm" etc.); a tag can also be given directly."""
        tags = DESCRIPTORS.get(descriptor)
        if tags is None:
            tags = [descriptor]
        cand = [e for e in self.good_entries() if e["tag"] in tags]
        if not cand:
            raise KeyError(f"no good snapshot for descriptor={descriptor!r}")
        f0_tgt = 440.0 * 2 ** ((pitch_midi - 69) / 12)
        best = min(cand, key=lambda e: abs(np.log2(max(e["f0_ref"], 1.0) / f0_tgt)))
        return load_snapshot(self.root / best["file"]), best

    def matrix(self):
        """Return (X[N,D] flattened snapshot parameters, meta[N]) for training a generative prior (parameter space, not waveforms)."""
        X, meta = [], []
        for e in self.index:
            snap = load_snapshot(self.root / e["file"])
            X.append(_flatten_snapshot(snap))
            meta.append(e)
        return np.asarray(X, dtype=np.float32), meta
def main(argv=None):
    """Build the labelled snapshot corpus: python -m library --tinysol <root> --nsynth <root> --out <dir>"""
    import argparse
    import sys
    from pathlib import Path as _P

    sys.path.insert(0, str(_P(__file__).resolve().parent))
    from eval.fidelity import nsynth_groups, tinysol_groups

    ap = argparse.ArgumentParser(description="Build the SnapshotLibrary corpus")
    ap.add_argument("--out", default="../../datasets/quill_corpus")
    ap.add_argument("--tinysol", default=None)
    ap.add_argument("--nsynth", default=None)
    ap.add_argument("--max-pitch", type=int, default=15)
    ap.add_argument("--per-family", type=int, default=4)
    args = ap.parse_args(argv)

    groups = []
    if args.tinysol:
        groups += tinysol_groups(args.tinysol, max_pitch=args.max_pitch, min_pitch=3)
    if args.nsynth:
        groups += nsynth_groups(
            args.nsynth, per_family=args.per_family, max_pitch=args.max_pitch, min_pitch=3
        )
    print(f"to analyse: {len(groups)} instruments -> {args.out}")
    idx = build_corpus(groups, args.out, source="dataset")
    from collections import Counter

    print("distribution by tag:", dict(Counter(e["tag"] for e in idx)))
    return 0


def _flatten_snapshot(snap, n_harm=60, n_mel=40, t_steady=1):
    """Compress a snapshot into a fixed-length vector (steady harmonic log amplitudes + noise band log means + log f0_ref) for prior modelling."""
    H = np.asarray(snap["H_env"], np.float32)  # [K, T]
    N = np.asarray(snap["N_env"], np.float32)  # [M, T]
    h = np.log(np.maximum(H.mean(axis=1), 1e-6))[:n_harm]
    h = np.pad(h, (0, max(0, n_harm - len(h))))[:n_harm]
    nz = np.log(np.maximum(N.mean(axis=1), 1e-6))[:n_mel]
    nz = np.pad(nz, (0, max(0, n_mel - len(nz))))[:n_mel]
    f0 = np.array([np.log(max(snap["f0_ref"], 1.0))], np.float32)
    return np.concatenate([h, nz, f0]).astype(np.float32)


def reconstruct_snapshot(flat, skeleton, n_harm=60, n_mel=40):
    """Inverse of _flatten_snapshot: graft a generated steady spectral shape onto the time dynamics and attack residual of a retrieved skeleton.

    The generative prior only produces a steady cross-section (101 dims), with no time evolution or attack; this function recolours the skeleton's
    time-varying envelope to the target steady spectrum per harmonic/noise band, keeping the skeleton's attack, decay and residual grain.
    Returns a renderable snapshot.
    """
    flat = np.asarray(flat, dtype=np.float64)
    h_t = np.exp(flat[:n_harm])  # target steady harmonic amplitudes
    n_t = np.exp(flat[n_harm : n_harm + n_mel])  # target steady noise bands
    H = np.asarray(skeleton["H_env"], np.float64).copy()
    N = np.asarray(skeleton["N_env"], np.float64).copy()
    K = min(n_harm, H.shape[0])
    M = min(n_mel, N.shape[0])
    sk_h = H[:K].mean(axis=1)  # skeleton steady cross-section
    sk_n = N[:M].mean(axis=1)
    H[:K] *= (h_t[:K] / (sk_h + 1e-8))[:, None]  # recolour per harmonic, keeping the time shape
    N[:M] *= (n_t[:M] / (sk_n + 1e-8))[:, None]
    snap = dict(skeleton)  # reuse attack_res/f0_ref/centers/partials
    snap["H_env"] = H.astype(np.float32)
    snap["N_env"] = N.astype(np.float32)
    snap["meta"] = dict(skeleton.get("meta", {}))
    snap["meta"]["name"] = "generated"
    snap["meta"]["skeleton"] = skeleton.get("meta", {}).get("name", "?")
    return snap


if __name__ == "__main__":
    raise SystemExit(main())
