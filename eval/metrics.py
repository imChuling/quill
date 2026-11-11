"""Evaluation toolkit: FR-13 (one-click regeneration of report/competition figures).

* confusion_matrix / classification_report: macro F1 + per-class P/R/F1 (articulation classification criteria)
* plot_confusion: confusion matrix heatmap PNG
* latency_decomposition: end-to-end latency decomposition table (MIDI->control->render compute->buffer->device)
* multiscale_spectral_loss: MSS (timbre fidelity, the same criterion across stages)

Depends on numpy + matplotlib (available on the eval side); shares the same metric definitions with the IDMT/MERT evaluation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ARTIC_LABELS = ["pluck", "slide", "legato", "strum", "sustain"]


# --------------------------------------------------------------------------- #
# Classification metrics
# --------------------------------------------------------------------------- #
def confusion_matrix(true, pred, labels=ARTIC_LABELS):
    idx = {l: i for i, l in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(true, pred):
        if t in idx and p in idx:
            M[idx[t], idx[p]] += 1
    return M


def classification_report(true, pred, labels=ARTIC_LABELS):
    """Return {label: {precision,recall,f1,support}, 'macro_f1':..., 'accuracy':...}."""
    M = confusion_matrix(true, pred, labels)
    out = {}
    f1s = []
    for i, l in enumerate(labels):
        tp = M[i, i]
        fp = M[:, i].sum() - tp
        fn = M[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[l] = {"precision": prec, "recall": rec, "f1": f1, "support": int(M[i, :].sum())}
        if M[i, :].sum() > 0:  # only classes that occur count toward the macro average
            f1s.append(f1)
    out["macro_f1"] = float(np.mean(f1s)) if f1s else 0.0
    out["accuracy"] = float(np.trace(M) / M.sum()) if M.sum() else 0.0
    return out


def plot_confusion(true, pred, path, labels=ARTIC_LABELS, title="Confusion matrix"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    M = confusion_matrix(true, pred, labels)
    Mn = M / np.maximum(M.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    rep = classification_report(true, pred, labels)
    ax.set_title(f"{title}\nmacro-F1={rep['macro_f1']:.3f}  acc={rep['accuracy']:.3f}")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{M[i, j]}",
                ha="center",
                va="center",
                color="white" if Mn[i, j] > 0.5 else "black",
                fontsize=9,
            )
    fig.colorbar(im, ax=ax, label="row-normalized")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return rep


# --------------------------------------------------------------------------- #
# Latency decomposition table (NFR-1: end to end <30ms)
# --------------------------------------------------------------------------- #
def measure_render_compute_ms(engine, n_voices=16, n_blocks=400):
    """Render compute time per block at full load; gc.disable reproduces performance conditions."""
    import gc
    import time

    from runtime.engine import Event

    bs = engine.blocksize
    engine.reset()
    for i in range(n_voices):
        engine._apply_event(Event("on", 45 + i, 110, 0.0))
    out = np.zeros(bs, dtype=np.float32)
    for _ in range(20):
        engine.process_block(out, bs, 0)
    gc.collect()
    try:
        gc.freeze()
    except Exception:
        pass
    gc.disable()
    try:
        ts = np.empty(n_blocks)
        for b in range(n_blocks):
            t0 = time.perf_counter()
            engine.process_block(out, bs, b * bs)
            ts[b] = (time.perf_counter() - t0) * 1000.0
    finally:
        gc.enable()
    return float(np.median(ts)), float(np.percentile(ts, 95))


def latency_decomposition(engine, device_output_ms=None, midi_jitter_ms=1.0):
    """End-to-end latency decomposition table (MIDI->control->render->buffer->device).

    Measurable locally: render compute, buffer budget; needs hardware measurement: device output latency (pass device_output_ms,
    or provided by engine.stream_latency_ms()). Returns (rows[(component, ms, note)], total_ms).
    """
    sr = engine.sr
    bs = engine.blocksize
    buffer_ms = bs / sr * 1000.0
    med_ms, p95_ms = measure_render_compute_ms(engine)
    ctrl_ms = 1000.0 / 250.0  # 250Hz ControlFrame granularity (in-block ramps)
    dev_ms = device_output_ms if device_output_ms is not None else engine.stream_latency_ms()
    rows = [
        ("MIDI input jitter", midi_jitter_ms, "queue->block start, worst case one block"),
        ("control update", ctrl_ms, "250Hz ControlFrame / in-block linear ramp"),
        ("render compute (16 voices)", p95_ms, f"p95; median {med_ms:.2f}ms (full load)"),
        ("output buffer", buffer_ms, f"blocksize {bs} / {sr}Hz"),
        ("device output", dev_ms, "measured on sound card (loopback / sd stream.latency)"),
    ]
    total = sum(r[1] for r in rows if np.isfinite(r[1]))
    return rows, float(total)


def format_latency_table(rows, total):
    lines = ["Latency decomposition (MIDI->DAC):", f"  {'component':<16}{'ms':>8}  note"]
    for name, ms, note in rows:
        v = f"{ms:8.2f}" if np.isfinite(ms) else "    n/a "
        lines.append(f"  {name:<16}{v}  {note}")
    lines.append(f"  {'total':<16}{total:8.2f}  (NFR-1 target <30ms; the device buffer dominates)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Multi-scale spectral loss (MSS, timbre fidelity)
# --------------------------------------------------------------------------- #
def multiscale_spectral_loss(y, y_re, sr=44100, ffts=(512, 1024, 2048, 4096)):
    n = min(len(y), len(y_re))
    y, y_re = y[:n], y_re[:n]
    losses = []
    for nfft in ffts:
        hop = nfft // 4
        S1 = _logmag(y, nfft, hop)
        S2 = _logmag(y_re, nfft, hop)
        m = min(S1.shape[1], S2.shape[1])
        losses.append(float(np.mean(np.abs(S1[:, :m] - S2[:, :m]))))
    return float(np.mean(losses))


def _logmag(y, nfft, hop):
    w = np.hanning(nfft)
    frames = [np.abs(np.fft.rfft(y[i : i + nfft] * w)) for i in range(0, len(y) - nfft, hop)]
    if not frames:
        frames = [np.abs(np.fft.rfft(np.zeros(nfft)))]
    return np.log(np.array(frames).T + 1e-5)
