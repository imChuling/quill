"""桌面"烘焙":一段取色录音 → 从预训练底座微调出一个 DDSP 音色(子进程,MPS)。

L0 取色秒级、实时,是默认路径;DDSP 能学到摩擦噪声、幅度起伏这类单条稳态谱抓不到的
质感,但要分钟级训练,所以做成取色之后可选的后台任务。只做桌面,不做网页(网页渲染
容器无 GPU)。

  python -m neural.bake_ddsp --wav take.wav --name myglass --base glass --steps 1500 \
      --out assets/user_ddsp --status assets/user_ddsp/_bake_status.json

产出 <out>/<name>.pt + <name>.json(底座/包络/步数/val)。NeuralTake 以 "ddsp:<name>"
加载,行为库缺省用底座的轨迹库(或同名 user_styles/<name>.npz)。
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

MIN_TAKE_S = 8.0  # 再短数据不够切验证集;20s 以下 UI 会提示可能过拟合


def chunk_dataset(y, sr_in, chunk_s=3.0, hop_s=1.5, min_s=0.9):
    """整段录音 → 重叠切片当"音"(train() 按音随机裁剪增强 + 切验证集)。

    特征(PESTO f0 / log-RMS)对整段只提一次再按帧切,比逐片提快一个量级;
    loudness 的标准化因此按整段算,与训练口径一致。
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
        tmp.replace(status_p)  # 原子写,UI 轮询不会读到半截

    try:
        write("preparing", step=0)
        # 取色 take 本身 ≤60 s;CLI 喂长文件时多给一些
        y, sr = load(str(wav), sr=44100, max_seconds=120.0)
        take_s = len(y) / sr
        if take_s < MIN_TAKE_S:
            write("failed", msg=f"take {take_s:.1f}s < {MIN_TAKE_S:.0f}s")
            return 1
        # PESTO 音高跟踪约 1× 实时(CPU),是 preparing 阶段的主要耗时,UI 据此给预期
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
        # 子进程里的失败要落到状态文件,UI 才看得见
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
