"""MERT 探针 —— FR-6 前沿升级（M8，阶段8 提前）。

依据 **MERTech, ICASSP 2024**：音乐 SSL 基座（HuggingFace `m-a-p/MERT`）抽 embedding +
轻量分类头做演奏法检测的 SOTA 范式。我们**只训轻量头，零自预训练**。

流程：事件音频段 → 24kHz → MERT 13 层 hidden states → 时间均值 + 层聚合 → 轻量头分类。
报告 ablation 三方对照：规则树 vs 随机森林 vs MERT 探针。

依赖 transformers + torch + nnAudio（已装）；权重首次自动下载（~95M 参数）。
引擎铁律 #12 不受影响：MERT 只在 analysis 侧、非实时路径。
"""
from __future__ import annotations

import numpy as np

_MERT_NAME = "m-a-p/MERT-v1-95M"
_MERT_SR = 24000
_model = None
_proc = None


def _load_mert():
    """惰性单例加载 MERT（重，加载一次）。"""
    global _model, _proc
    if _model is None:
        import torch  # noqa
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        _model = AutoModel.from_pretrained(_MERT_NAME, trust_remote_code=True).eval()
        _proc = Wav2Vec2FeatureExtractor.from_pretrained(_MERT_NAME,
                                                         trust_remote_code=True)
    return _model, _proc


def mert_embed(y, sr, layers=None):
    """音频 -> MERT 嵌入向量。

    layers=None: 13 层时间均值再做层均值 -> 768 维（紧凑，少样本不易过拟合）；
    layers=list: 仅取指定层（MERTech 发现中层对技巧检测更优，可调）后拼接。
    """
    import torch
    import librosa
    model, proc = _load_mert()
    y = np.asarray(y, dtype=np.float32)
    if sr != _MERT_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=_MERT_SR)
    if len(y) < _MERT_SR // 10:                 # 太短补够 ~0.1s
        y = np.pad(y, (0, _MERT_SR // 10 - len(y)))
    inp = proc(y, sampling_rate=_MERT_SR, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    hs = torch.stack(out.hidden_states)         # [13, 1, T, 768]
    tmean = hs.mean(dim=2).squeeze(1)           # [13, 768] 时间均值
    if layers is None:
        emb = tmean.mean(dim=0)                  # 层均值 -> [768]
    else:
        emb = tmean[layers].reshape(-1)          # 选层拼接
    return emb.cpu().numpy().astype(np.float32)



# --------------------------------------------------------------------------- #
# 轻量分类头（MERTech：基座冻结，只训头；秒-分钟级）
# --------------------------------------------------------------------------- #
def train_probe(X, labels, kind="logreg", seed=0):
    """X[n,dim] + labels -> 轻量头分类器（标准化 + LogReg/线性 SVM）。"""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                                 random_state=seed)
    else:
        from sklearn.svm import LinearSVC
        clf = LinearSVC(C=1.0, class_weight="balanced", random_state=seed)
    pipe = make_pipeline(StandardScaler(), clf)
    pipe.fit(X, labels)
    return pipe


def probe_predict(pipe, X):
    return list(pipe.predict(X))
