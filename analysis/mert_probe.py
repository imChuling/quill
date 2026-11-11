"""MERT probe: extract embeddings from a music SSL backbone (HuggingFace `m-a-p/MERT`) + a lightweight head for articulation classification.

MERTech (ICASSP 2024) paradigm: frozen backbone, only a lightweight head is trained.
Pipeline: event audio segment -> 24kHz -> MERT 13-layer hidden states -> time mean + layer aggregation -> lightweight head classification.

Depends on transformers + torch + nnAudio; weights are downloaded automatically on first use (~95M parameters).
Used only on the analysis side, off the real-time path.
"""

from __future__ import annotations

import numpy as np

_MERT_NAME = "m-a-p/MERT-v1-95M"
_MERT_SR = 24000
_model = None
_proc = None


def _load_mert():
    """Lazily load MERT as a singleton (heavy, loaded once)."""
    global _model, _proc
    if _model is None:
        import torch  # noqa
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        _model = AutoModel.from_pretrained(_MERT_NAME, trust_remote_code=True).eval()
        _proc = Wav2Vec2FeatureExtractor.from_pretrained(_MERT_NAME, trust_remote_code=True)
    return _model, _proc


def mert_embed(y, sr, layers=None):
    """Audio -> MERT embedding vector.

    layers=None: time mean of 13 layers then layer mean -> 768 dims (compact, less overfitting with few samples);
    layers=list: take only the given layers (MERTech found middle layers better for technique detection, tunable) and concatenate.
    """
    import librosa
    import torch

    model, proc = _load_mert()
    y = np.asarray(y, dtype=np.float32)
    if sr != _MERT_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=_MERT_SR)
    if len(y) < _MERT_SR // 10:  # pad too-short input to ~0.1s
        y = np.pad(y, (0, _MERT_SR // 10 - len(y)))
    inp = proc(y, sampling_rate=_MERT_SR, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    hs = torch.stack(out.hidden_states)  # [13, 1, T, 768]
    tmean = hs.mean(dim=2).squeeze(1)  # [13, 768] time mean
    emb = tmean.mean(dim=0) if layers is None else tmean[layers].reshape(-1)
    return emb.cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------- #
# Lightweight classification head (frozen backbone, only the head is trained)
# --------------------------------------------------------------------------- #
def train_probe(X, labels, kind="logreg", seed=0):
    """X[n,dim] + labels -> lightweight head classifier (standardisation + LogReg/linear SVM)."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=seed)
    else:
        from sklearn.svm import LinearSVC

        clf = LinearSVC(C=1.0, class_weight="balanced", random_state=seed)
    pipe = make_pipeline(StandardScaler(), clf)
    pipe.fit(X, labels)
    return pipe


def probe_predict(pipe, X):
    return list(pipe.predict(X))
