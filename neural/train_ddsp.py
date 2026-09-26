"""DDSP 多音泛化训练（单件音源纯净乐器）—— L2 真训练，非单音过拟合。

给一件**具体乐器**（音源纯净：电贝斯 / clean 电吉他 / 木吉他）的多音，学 (f0,loudness)→音色
的映射 → 一个能被演奏法层 ControlFrame 驱动、跨音域发声的神经音色。失真等留作下游 FX。

跑：python -W ignore -m neural.train_ddsp --glob "<dir>/*.wav" --name electric_bass --steps 4000
MPS 本地先验证，Modal 放大。特征缓存到 .npz 避免重复 PESTO。
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural.ddsp import DDSP, multiscale_spectral_loss, extract_features, SR, HOP  # noqa: E402
from analysis.io import load  # noqa: E402


def _trim_silence(y, sr, thresh_db=-40.0, pad_s=0.03):
    """裁掉首尾静音/低能量段（静音帧会教会 DDSP 噪声分支学底噪）。"""
    y = np.asarray(y, dtype=np.float32)
    e = np.abs(y)
    pk = float(e.max())
    if pk <= 0:
        return y
    above = np.where(e > pk * 10 ** (thresh_db / 20.0))[0]
    if above.size < 2:
        return y
    s = max(0, above[0] - int(pad_s * sr))
    t = min(len(y), above[-1] + int(pad_s * 2 * sr))       # 尾部多留一点衰减
    return y[s:t]


def build_dataset(files, sr_in=44100, cache=None, min_s=0.9, verbose=True, trim=True):
    """[(f0[1,T,1], loud[1,T,1], audio[1,T*HOP])] **整段**（变长，训练时随机裁剪增强）。"""
    if cache and Path(cache).exists():
        d = np.load(cache)
        n = int(d["n"])
        out = [(torch.tensor(d[f"f0_{i}"]), torch.tensor(d[f"lo_{i}"]),
                torch.tensor(d[f"au_{i}"])) for i in range(n)]
        if verbose:
            print(f"  载入缓存 {n} 音 <- {cache}")
        return out
    Tmin = int(min_s * SR / HOP)
    out = []
    for k, f in enumerate(files):
        try:
            y, _ = load(f, sr=sr_in)
            if trim:
                y = _trim_silence(y, sr_in)                # 数据清洗：裁静音
            f0, loud, au = extract_features(y, sr_in)
        except Exception:
            continue
        if f0.shape[1] < Tmin:
            continue
        out.append((f0, loud, au))
        if verbose and (k + 1) % 25 == 0:
            print(f"  特征 {k+1}/{len(files)} ...")
    if cache:
        save = {"n": len(out)}
        for i, (a, b, c) in enumerate(out):
            save[f"f0_{i}"] = a.numpy(); save[f"lo_{i}"] = b.numpy(); save[f"au_{i}"] = c.numpy()
        np.savez(cache, **save)
    return out


def _crop_batch(data, idx, Tc, rng, device, rand=True):
    """对一批音随机裁 Tc 帧窗口（增强：每步看不同片段）。"""
    f0s, los, aus = [], [], []
    for i in idx:
        f0, lo, au = data[i]
        T = f0.shape[1]
        s = int(rng.integers(0, T - Tc + 1)) if (rand and T > Tc) else 0
        f0s.append(f0[:, s:s + Tc]); los.append(lo[:, s:s + Tc])
        aus.append(au[:, s * HOP:(s + Tc) * HOP])
    return (torch.cat(f0s).to(device), torch.cat(los).to(device), torch.cat(aus).to(device))


def train(data, steps=5000, batch=8, lr=1e-3, weight_decay=1e-4, dropout=0.1,
          crop_s=1.0, device=None, val=0.15, verbose=True,
          ckpt_path=None, init_state=None, temporal="gru", n_layers=2):
    """随机裁剪增强 + weight decay + dropout + **早停** + 断点续训(被 kill 可续)。"""
    import copy
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    rng = np.random.default_rng(0)
    Tc = int(crop_s * SR / HOP)
    n = len(data)
    perm = rng.permutation(n)
    nval = max(1, int(n * val))
    vi, ti = perm[:nval], perm[nval:]
    model = DDSP(dropout=dropout, temporal=temporal, n_layers=n_layers).to(device)
    if init_state is not None:
        model.load_state_dict(init_state); print("  从断点续训")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    losses, best_val, best_state = [], float("inf"), None
    for it in range(steps):
        model.train()
        idx = ti[rng.integers(0, len(ti), batch)]
        f0, lo, au = _crop_batch(data, idx, Tc, rng, device, rand=True)
        opt.zero_grad()
        yhat = model(f0, lo)
        m = min(yhat.shape[1], au.shape[1])
        loss = multiscale_spectral_loss(yhat[:, :m], au[:, :m])
        loss.backward(); opt.step(); sched.step()
        losses.append(float(loss.item()))
        if (it + 1) % 250 == 0:
            model.eval()
            with torch.no_grad():
                f0v, lov, auv = _crop_batch(data, vi, Tc, rng, device, rand=False)
                yv = model(f0v, lov)
                m = min(yv.shape[1], auv.shape[1])
                vl = float(multiscale_spectral_loss(yv[:, :m], auv[:, :m]).item())
            if vl < best_val:                              # 早停：记最优 val 的权重
                best_val = vl
                best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
                if ckpt_path:                              # 断点：被 kill 也能续
                    torch.save(best_state, ckpt_path)
            if verbose:
                print(f"  step {it+1:5d}  train≈{np.mean(losses[-100:]):.2f}  val={vl:.2f}"
                      f"  best={best_val:.2f}")
    if best_state is not None:
        model.load_state_dict(best_state)                  # 回到 val 最优
    return model, losses, best_val, vi


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="一件乐器的音 glob")
    ap.add_argument("--name", default="instrument")
    ap.add_argument("--max-notes", type=int, default=120)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--out", default="assets/demos")
    args = ap.parse_args(argv)

    import soundfile as sf
    files = sorted(glob.glob(args.glob, recursive=True))[:args.max_notes]
    print(f"{args.name}: {len(files)} 音 -> 特征中…")
    cache = f"/tmp/ddsp_{args.name}_full.npz"               # 整段特征(新格式)
    data = build_dataset(files, cache=cache)
    print(f"  数据集 {len(data)} 音；随机裁剪增强 + 早停 + wd + dropout + 断点续训，训 {args.steps} 步")
    ckpt = f"/tmp/ddsp_{args.name}.ckpt"
    init = torch.load(ckpt, map_location="cpu") if Path(ckpt).exists() else None
    model, losses, best_val, vi = train(data, steps=args.steps, ckpt_path=ckpt, init_state=init)
    Path(ckpt).unlink(missing_ok=True)
    dev = next(model.parameters()).device
    Path(args.out).mkdir(parents=True, exist_ok=True)
    # 重建 3 个**持出**音整段（泛化检验）
    with torch.no_grad():
        for j, i in enumerate(vi[:3]):
            f0, lo, au = data[i]
            yhat = model(f0.to(dev), lo.to(dev)).cpu().numpy()[0]
            orig = au.numpy()[0]
            sf.write(f"{args.out}/ddsp_{args.name}_val{j}_orig.wav",
                     (orig/(np.max(np.abs(orig))+1e-9)*0.9).astype(np.float32), SR)
            sf.write(f"{args.out}/ddsp_{args.name}_val{j}_recon.wav",
                     (yhat/(np.max(np.abs(yhat))+1e-9)*0.9).astype(np.float32), SR)
    torch.save(model.state_dict(), f"{args.out}/ddsp_{args.name}.pt")
    print(f"  train≈{np.mean(losses[-100:]):.2f}  best_val={best_val:.2f}；持出重建 -> ddsp_{args.name}_val*_recon.wav")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
