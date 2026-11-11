"""神经渲染 take:卷帘音符(绝对时间) → 行为检索 + DDSP → 整段音频。

app 与神经管线的适配层:
  - 卷帘 notes [{midi,start,len,vel,art}] → 按间隙切乐句(phrase)
  - 每个乐句转成 control_from_behavior 的顺序 score [{midi,dur,art}]
  - 逐乐句渲染,再按绝对 start 摆回时间线

乐器注册表 INSTRUMENTS:每个乐器 = 轨迹库 + DDSP checkpoint + 包络/滑音策略。
  violin   URMP 小提琴,7 维特征(带演奏法标签,条件化检索),sustain 包络
  trumpet  URMP 小号,sustain 包络
  guitar   GuitarSet,6 维特征(无演奏法标签,普通最近邻),pluck 包络
  glass    rubbed-glass:DDSP 用 82 s 录音训,轨迹库 = 24 音 style 库
桌面版与网页版(ui/render_service.py)共用这张表,两边音色集合一致。

演奏法映射(上游把 art 字符串填好即可):
  "vibrato" -> 1   "slide" -> 2   其余("pluck"/"legato"/缺省) -> 0 plain
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

QUILL_DIR = Path(__file__).resolve().parent.parent
CKPT_DIR = QUILL_DIR / "checkpoints"
USER_STYLES_DIR = QUILL_DIR / "assets" / "user_styles"

ART_MAP = {"vibrato": 1, "slide": 2}
PHRASE_GAP = 0.08  # 音符间隙超过 80 ms 视为乐句边界(之内算 legato 连奏)
EDGE_FADE_S = 0.005  # 乐句边缘淡入淡出,防拼接咔哒
SLIDE_RANGE_CAP = 400.0  # slide 轨迹 pitch 超此 cents 视为离群(修八度跳)

INSTRUMENTS = {
    "violin": {
        "traj": CKPT_DIR / "urmp_vn_trajectories.npz",
        "ckpt": CKPT_DIR / "ddsp_urmp_vn_12k.pt",
        "envelope": "sustain",
    },
    "trumpet": {
        "traj": CKPT_DIR / "urmp_tpt_trajectories.npz",
        "ckpt": CKPT_DIR / "ddsp_urmp_tpt_12k.pt",
        "envelope": "sustain",
    },
    "guitar": {
        "traj": CKPT_DIR / "guitarset_trajectories.npz",
        "ckpt": CKPT_DIR / "ddsp_guitarset.pt",
        "envelope": "pluck",
    },
    "glass": {
        "traj": CKPT_DIR / "glass_style.npz",
        "ckpt": CKPT_DIR / "ddsp_glass.pt",
        "envelope": "sustain",
    },
}

# 仅行为源(有轨迹库、无 DDSP 音色):可作 behavior,不可作 instrument。
BEHAVIOR_ONLY: dict = {}

# vibrato 模型的 instrument one-hot 槽位按 tools/train_vibrato.py 的训练顺序
# (violin=0, cello=1, guitar=2),与 INSTRUMENTS 的字典顺序无关。未训过的音色落到
# violin 槽(同为持续类乐器)。
VIBRATO_INST_IDX = {"violin": 0, "guitar": 2}


# UI 层(桌面卷帘/网页)的默认演奏参数。不做成 render() 的默认值,是为了让 eval 脚本
# 保持 λ=0 / timing=0 的基线口径。
UI_RENDER_DEFAULTS = {"timing": 0.5, "continuity": 1.0, "contiguity": 1.0, "motion": 1.0}


def available_instruments():
    """checkpoint 与轨迹库都在本机的音色名(UI 列表与服务端 /status 共用)。"""
    return [k for k, s in INSTRUMENTS.items() if s["ckpt"].exists() and s["traj"].exists()]


# 桌面烘焙出的用户 DDSP 音色(neural/bake_ddsp.py):<name>.pt + <name>.json
USER_DDSP_DIR = QUILL_DIR / "assets" / "user_ddsp"


def user_ddsp_instruments():
    """"ddsp:<name>" 列表(有 .pt 且 .json 可读)。"""
    import json

    out = []
    if USER_DDSP_DIR.is_dir():
        for p in sorted(USER_DDSP_DIR.glob("*.pt")):
            if p.name.endswith(".partial.pt") or not p.with_suffix(".json").exists():
                continue
            try:
                json.loads(p.with_suffix(".json").read_text())
                out.append(f"ddsp:{p.stem}")
            except Exception:
                continue
    return out


def _user_ddsp_spec(name):
    """烘焙音色的 _Voice 规格:自己的 .pt;轨迹库 = 同名 user_styles 库(若有)否则底座的。"""
    import json

    meta = json.loads((USER_DDSP_DIR / f"{name}.json").read_text())
    base = INSTRUMENTS.get(meta.get("base", "violin"), INSTRUMENTS["violin"])
    own = USER_STYLES_DIR / f"{name}.npz"  # 保存音色/烘焙用同一条命名规则(bridge._safe_name)
    return {
        "ckpt": USER_DDSP_DIR / f"{name}.pt",
        "traj": own if own.exists() else base["traj"],
        "envelope": meta.get("envelope", base["envelope"]),
    }

# 学习型行为源仍只提供 behavior,音色继续由 instrument 的 Quill DDSP 解码器提供。
# 文件由 tools/train_distilled_behavior.py 生成;不存在时不出现在 UI 列表中。
DISTILLED_BEHAVIORS = {
    "mididdsp_distilled": CKPT_DIR / "behavior_distilled.pt",
}


class ConditionedNN:
    """演奏法条件化最近邻。

    7 维特征(最后一维=art)时条件化检索;6 维特征(库未标演奏法)时退化为普通最近邻
    (use_art=False)。timing 表 (N,2):每条 entry 的 [articulation ratio, agogic ratio],
    从 features 的 dur/next_gap(和可选 meta 的 track 分组)派生,不需要重新提取库。
    predict_timing() 与轨迹检索同池。
    """

    def __init__(self, features, trajectories, slide_cap=SLIDE_RANGE_CAP, meta=None):
        self.features = np.asarray(features, dtype=np.float32)
        self.trajectories = np.asarray(trajectories, dtype=np.float32)
        self.use_art = self.features.shape[1] >= 7
        if self.use_art:
            art = self.features[:, 6].astype(np.int64)
            # 离群过滤:slide 池里 pitch range 过大的轨迹(库里有 >700 c 的极端滑音,即八度跳)
            rng = self.trajectories[:, 0].max(axis=1) - self.trajectories[:, 0].min(axis=1)
            keep = (art != 2) | (rng <= slide_cap)
            self.features, self.trajectories, art = (
                self.features[keep],
                self.trajectories[keep],
                art[keep],
            )
            if meta is not None and len(meta) == len(keep):
                meta = [meta[i] for i in np.where(keep)[0]]
            self.art = art
            self._pools = {a: np.where(art == a)[0] for a in np.unique(art)}
        self.base = self.features[:, :6]
        self._std = self.base.std(axis=0) + 1e-8
        self._norm = self.base / self._std
        self.timing = self._compute_timing(self.features, meta)
        # 连续检索索引:(track, note_idx) → 库索引,predict(contiguity>0) 用它找"上一音的下一音"
        self.meta = meta if (meta is not None and len(meta) == len(self.features)) else None
        self.last_idx = None
        self._contig_index = {}
        if self.meta:
            for i in range(len(self.meta)):
                k = self._contig_key(i)
                if k is not None:
                    self._contig_index[k] = i

    supports_concat = True  # predict() 接受 prev_traj/concat_lambda/contiguity(蒸馏模型不接受)

    def clone(self):
        """共享全部数组、只有自己 last_idx 的浅副本。

        实时弹奏(MIDI 线程)与离线渲染(UI 线程)各拿一个,连续检索的"上一音"记忆互不串。
        """
        import copy

        c = copy.copy(self)
        c.last_idx = None
        return c

    @staticmethod
    def _compute_timing(features, meta):
        """每条 entry 的 [articulation ratio, agogic]。

        articulation ratio = 发声时长 / IOI(IOI = dur + next_gap):连奏≈1,断奏<1。
        guitar 库有余音重叠(dur>IOI),钳到 1;next_gap>0.5 视为句尾,不缩短。
        agogic = 本音 IOI / 同曲邻域 IOI 中位(需要 meta 的 track 分组):
        rubato 式局部伸缩,钳到 0.7–1.4;无 meta 时恒 1(退化为只有 articulation)。
        """
        dur = features[:, 1].astype(np.float64)
        nxt = features[:, 3].astype(np.float64)
        ioi = dur + np.clip(nxt, 0.0, None)
        art_ratio = np.clip(dur / np.maximum(ioi, 1e-6), 0.25, 1.0)
        art_ratio[nxt > 0.5] = 1.0
        agogic = np.ones(len(dur), np.float64)
        if meta:
            track_of = [
                str(m.get("track") or m.get("track_id") or m.get("src") or "") for m in meta
            ]
            by_track = {}
            for i, t in enumerate(track_of):
                by_track.setdefault(t, []).append(i)
            W = 4  # 邻域半径(±4 音 → 9 音窗口)
            for idxs in by_track.values():
                seq_ioi = ioi[idxs]
                for j, i in enumerate(idxs):
                    if nxt[i] > 0.5:
                        continue  # 句尾 IOI 无意义,保持 1
                    lo, hi = max(0, j - W), min(len(idxs), j + W + 1)
                    local = float(np.median(seq_ioi[lo:hi]))
                    if local > 1e-6:
                        agogic[i] = np.clip(ioi[i] / local, 0.7, 1.4)
        return np.stack([art_ratio, agogic], axis=1).astype(np.float32)

    _CONCAT_EDGE = 4  # 端点取几帧均值算拼接代价

    def _pool_and_dist(self, feat):
        """给定特征向量,返回 (pool indices, context distances)。"""
        feat = np.asarray(feat, dtype=np.float32)
        if self.use_art:
            a = int(round(float(feat[6]))) if len(feat) >= 7 else 0
            pool = self._pools.get(a)
            if pool is None or len(pool) == 0:
                pool = np.arange(len(self.base))
        else:
            pool = np.arange(len(self.base))
        q = feat[:6] / self._std
        dist = np.linalg.norm(self._norm[pool] - q, axis=1)
        return pool, dist

    def _nearest(self, feat):
        pool, dist = self._pool_and_dist(feat)
        return int(pool[int(np.argmin(dist))])

    def _contig_key(self, idx):
        """(track, note_idx) 供"同一源乐句的下一个音"判定;无 meta 时 None。"""
        m = self.meta[idx] if self.meta and idx < len(self.meta) else None
        if not m:
            return None
        tr = m.get("track") or m.get("track_id") or m.get("src")
        ni = m.get("note_idx")
        return (str(tr), int(ni)) if tr is not None and ni is not None else None

    def predict(self, feat, prev_traj=None, concat_lambda=0.0, contiguity=0.0):
        """检索一条 (4,48) 轨迹。

        concat_lambda>0:在 top-8 里按与上一音端点的拼接代价重排序;contiguity>0:候选若
        正好是上一音所选源音的下一个音(同 track、note_idx+1),代价再减 contiguity,让整句
        尽量取自同一真实乐句。选中的库索引记在 self.last_idx。
        """
        last = getattr(self, "last_idx", None)
        if prev_traj is None or (concat_lambda <= 0 and contiguity <= 0):
            idx = self._nearest(feat)
            self.last_idx = idx
            return self.trajectories[idx]
        pool, dist = self._pool_and_dist(feat)
        k = min(8, len(pool))
        top_k = np.argpartition(dist, k)[:k] if k < len(pool) else np.arange(len(pool))
        e = self._CONCAT_EDGE
        prev_p = float(np.mean(prev_traj[0, -e:]))
        prev_e = float(np.mean(prev_traj[1, -e:]))
        want = None
        if contiguity > 0 and last is not None:
            pk = self._contig_key(last)
            if pk is not None:
                want = (pk[0], pk[1] + 1)
                # 下一个音不一定在 top-8 里:只要同标签池里有,就把它加进候选
                nxt = self._contig_index.get(want) if hasattr(self, "_contig_index") else None
                if nxt is not None and nxt in set(pool.tolist()) and nxt not in pool[top_k]:
                    j_extra = int(np.where(pool == nxt)[0][0])
                    top_k = np.append(top_k, j_extra)
        best_j, best_cost = top_k[0], float("inf")
        for j in top_k:
            idx = int(pool[j])
            cand = self.trajectories[idx]
            cc = abs(prev_p - float(np.mean(cand[0, :e]))) + abs(
                prev_e - float(np.mean(cand[1, :e]))
            )
            total = float(dist[j]) + concat_lambda * cc
            if want is not None and self._contig_key(idx) == want:
                total -= contiguity
            if total < best_cost:
                best_cost = total
                best_j = j
        self.last_idx = int(pool[best_j])
        return self.trajectories[self.last_idx]

    # timing 检索的距离掩码:掐掉 prev_gap/next_gap 两维。gap 是 timing 行为的输出,
    # 不是检索条件:按连奏 query 的 gap=0 检索只会检回连奏音符,timing 永远迁移不出来。
    _TIMING_MASK = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0], np.float32)

    def predict_timing(self, feat):
        """检索最近 entry 的 (articulation ratio, agogic ratio)。

        与轨迹检索同池,但距离只看 [midi, dur, 句首/句尾旗标]。
        """
        feat = np.asarray(feat, dtype=np.float32)
        if self.use_art:
            a = int(round(float(feat[6]))) if len(feat) >= 7 else 0
            pool = self._pools.get(a)
            if pool is None or len(pool) == 0:
                pool = np.arange(len(self.base))
        else:
            pool = np.arange(len(self.base))
        q = feat[:6] / self._std
        dist = np.linalg.norm((self._norm[pool] - q) * self._TIMING_MASK, axis=1)
        r, g = self.timing[pool[int(np.argmin(dist))]]
        return float(r), float(g)


TIMING_MAX_SHIFT = 0.08  # agogic onset 偏移上限(秒):微观 rubato,不动宏观节奏


def apply_timing(notes, behavior_model, strength=1.0):
    """卷帘音符表 → 带检索 timing 行为的新表(原表不改)。

    每个音符按与轨迹检索相同的特征找库里最近的源音符,取其 articulation ratio(缩短发声
    时长,在 IOI 内制造真实空隙,经 _phrases 的 80 ms 规则变成断奏起音)和 agogic ratio
    (onset 微偏移,钳 ±80 ms)。strength=0 原样返回;1 = 库里的 timing 原强度。
    behavior_model 无 predict_timing(如蒸馏模型)时原样返回。
    """
    if not notes or strength <= 0 or not hasattr(behavior_model, "predict_timing"):
        return notes
    seq = sorted(notes, key=lambda x: float(x["start"]))
    n = len(seq)
    out = []
    for i, nt in enumerate(seq):
        t0, ln = float(nt["start"]), float(nt["len"])
        ioi = (float(seq[i + 1]["start"]) - t0) if i < n - 1 else ln
        ioi = max(ioi, 0.05)
        prev_gap = (t0 - (float(seq[i - 1]["start"]) + float(seq[i - 1]["len"]))) if i > 0 else 1.0
        next_gap = (float(seq[i + 1]["start"]) - (t0 + ln)) if i < n - 1 else 1.0
        a = nt.get("art", 0)
        art_i = int(a) if isinstance(a, (int, float)) else ART_MAP.get(str(a), 0)
        feat = np.array(
            [
                float(nt["midi"]),
                ln,
                max(prev_gap, 0.0),
                max(next_gap, 0.0),
                1.0 if (i == 0 or prev_gap > 0.5) else 0.0,
                1.0 if (i == n - 1 or next_gap > 0.5) else 0.0,
                float(art_i),
            ],
            np.float32,
        )
        r, g = behavior_model.predict_timing(feat)
        shift = float(np.clip((g - 1.0) * ioi, -TIMING_MAX_SHIFT, TIMING_MAX_SHIFT)) * strength
        cur_ratio = min(ln / ioi, 1.0)
        mix = cur_ratio + (r - cur_ratio) * strength
        new_len = float(np.clip(ioi * mix, 0.06, ioi))
        nt2 = dict(nt)
        nt2["start"] = max(0.0, t0 + shift)
        nt2["len"] = new_len
        out.append(nt2)
    return out


def _load_meta(d):
    """npz 里的 meta(json 字符串)→ list;没有则 None。"""
    if "meta" not in getattr(d, "files", []):
        return None
    try:
        import json

        return json.loads(str(d["meta"]))
    except Exception:
        return None


class _Voice:
    """单个乐器:轨迹库 NN + DDSP 模型 + 包络策略。"""

    def __init__(self, spec, device):
        import torch

        from neural.ddsp import DDSP

        d = np.load(str(spec["traj"]))
        self.nn = ConditionedNN(d["features"], d["trajectories"], meta=_load_meta(d))
        self.envelope = spec["envelope"]
        self.model = DDSP(dropout=0.0).to(device)
        self.model.load_state_dict(
            torch.load(str(spec["ckpt"]), map_location=device, weights_only=True)
        )
        self.model.eval()


class NeuralTake:
    """按需加载乐器(首次 ~1s/个),之后 render() 复用。CPU 即可(~80x 实时)。"""

    def __init__(self, device=None):
        import threading

        import torch

        from neural.ddsp import SR

        self.sr = SR
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._voices = {}
        self._custom_nns = {}
        self._style_lock = threading.Lock()

    def voice(self, instrument="violin"):
        if str(instrument).startswith("ddsp:"):
            key = str(instrument)
            spec = _user_ddsp_spec(key[5:])
            mt = spec["ckpt"].stat().st_mtime  # 同名音色重新烘焙后要换权重
            if key not in self._voices or self._voices[key]._mtime != mt:
                v = _Voice(spec, self.device)
                v._mtime = mt
                self._voices[key] = v
            return self._voices[key]
        if instrument not in INSTRUMENTS:
            instrument = "violin"
        if instrument not in self._voices:
            self._voices[instrument] = _Voice(INSTRUMENTS[instrument], self.device)
        return self._voices[instrument]

    def _custom_nn(self, name):
        """加载仅行为源(BEHAVIOR_ONLY)或用户自定义库(user_styles)→ ConditionedNN。"""
        with self._style_lock:
            return self._custom_nn_unlocked(name)

    def _custom_nn_unlocked(self, name):
        if name not in self._custom_nns:
            if name in DISTILLED_BEHAVIORS:
                p = DISTILLED_BEHAVIORS[name]
                if not p.exists():
                    return None
                from neural.behavior_distill import DistilledBehavior

                self._custom_nns[name] = DistilledBehavior(p, device="cpu", residual_strength=1.5)
                return self._custom_nns[name]
            elif name in BEHAVIOR_ONLY:
                p = BEHAVIOR_ONLY[name]["traj"]
            else:
                p = USER_STYLES_DIR / f"{name}.npz"
            if not p.exists():
                return None
            d = np.load(str(p))
            self._custom_nns[name] = ConditionedNN(
                d["features"], d["trajectories"], meta=_load_meta(d)
            )
        return self._custom_nns[name]

    def resolve_behavior(self, behavior, timbre_voice=None, instrument_name=None):
        """行为源 → (行为模型, 包络策略)。render 与桌面实时行为模式共用这一个解析。

        behavior: 带 .predict 的对象 → 直接用;None/""/与音色同名 → 音色自己的轨迹库
        (无音色声部时,如 L0 取色音色,抛 ValueError);内置乐器名 → 其轨迹库;其余 →
        user_styles / 蒸馏模型,找不到时退回音色自己的库,退无可退则抛 ValueError。
        """
        if hasattr(behavior, "predict"):
            return behavior, "sustain"
        if behavior in (None, "") or behavior == instrument_name:
            if timbre_voice is None:
                raise ValueError("captured L0 timbre needs an explicit behavior library")
            return timbre_voice.nn, timbre_voice.envelope
        if behavior in INSTRUMENTS:
            v = self.voice(behavior)
            return v.nn, v.envelope
        custom = self._custom_nn(behavior)
        if custom is not None:
            return custom, "sustain"
        if timbre_voice is None:
            raise ValueError(f"unknown behavior library {behavior!r}")
        return timbre_voice.nn, timbre_voice.envelope

    def list_behavior_sources(self):
        """列出所有可用行为源:内置乐器 + 仅行为源 + 用户导入的自定义库。"""
        sources = []
        for k in INSTRUMENTS:
            sources.append(
                {
                    "name": k,
                    "builtin": True,
                    "notes": int(np.load(str(INSTRUMENTS[k]["traj"]))["features"].shape[0])
                    if INSTRUMENTS[k]["traj"].exists()
                    else 0,
                }
            )
        for k, spec in BEHAVIOR_ONLY.items():
            if spec["traj"].exists():
                sources.append(
                    {
                        "name": k,
                        "builtin": True,
                        "notes": int(np.load(str(spec["traj"]))["features"].shape[0]),
                    }
                )
        for k, path in DISTILLED_BEHAVIORS.items():
            if path.exists():
                import torch

                ck = torch.load(str(path), map_location="cpu", weights_only=True)
                n = int(ck["prototype_features"].shape[0])
                sources.append({"name": k, "builtin": True, "model": True, "notes": n})
        USER_STYLES_DIR.mkdir(parents=True, exist_ok=True)
        for f in sorted(USER_STYLES_DIR.glob("*.npz")):
            try:
                d = np.load(str(f))
                n = int(d["features"].shape[0])
            except Exception:
                n = 0
            sources.append({"name": f.stem, "builtin": False, "notes": n})
        return sources

    def import_style(self, npz_path):
        """导入一个 style.npz(extract_style.py 产出)到 user_styles,返回名称。"""
        p = Path(npz_path)
        if not p.exists():
            raise FileNotFoundError(npz_path)
        USER_STYLES_DIR.mkdir(parents=True, exist_ok=True)
        dst = USER_STYLES_DIR / p.name
        if dst != p:
            import shutil

            shutil.copy2(str(p), str(dst))
        name = dst.stem
        with self._style_lock:
            self._custom_nns.pop(name, None)
        return name

    def delete_style(self, name):
        """删除用户自定义行为库(名字只允许落在 USER_STYLES_DIR 内,防 ../ 逃逸)。"""
        nm = str(name or "")
        if not nm or "/" in nm or "\\" in nm or nm in (".", ".."):
            return
        p = (USER_STYLES_DIR / f"{nm}.npz").resolve()
        if p.parent != USER_STYLES_DIR.resolve():
            return
        if p.exists():
            p.unlink()
        with self._style_lock:
            self._custom_nns.pop(name, None)

    # ---- 卷帘 notes → 乐句列表 ----
    @staticmethod
    def _phrases(notes):
        seq = sorted(notes, key=lambda x: float(x["start"]))
        phrases, cur = [], []
        for nt in seq:
            t0, ln = float(nt["start"]), float(nt["len"])
            if cur:
                prev_end = cur[-1]["t0"] + cur[-1]["dur"]
                if t0 - prev_end > PHRASE_GAP:
                    phrases.append(cur)
                    cur = []
                else:
                    cur[-1]["dur"] = max(0.05, t0 - cur[-1]["t0"])  # 单音化:截到下一音起点
            cur.append(
                {
                    "t0": t0,
                    "dur": max(0.05, ln),
                    "midi": int(nt["midi"]),
                    "vel": int(nt.get("vel", 100)),
                    "art": ART_MAP.get(str(nt.get("art", "")), 0),
                }
            )
        if cur:
            phrases.append(cur)
        return phrases

    @staticmethod
    def _monophonize(notes):
        """同 onset 的和弦只保最高音(DAW region / 卷帘手画都会出现),引擎是单声部的。"""
        from analysis.midi_dialect import ImportReport, _monophonize

        events = [
            (
                float(n["start"]),
                float(n["len"]),
                int(n["midi"]),
                n,
                None,
            )
            for n in notes
        ]
        mono = _monophonize(events, ImportReport())
        out = []
        for onset, dur, _midi, src, _x in mono:
            nt = dict(src)
            nt["start"], nt["len"] = onset, dur
            out.append(nt)
        return out

    def render(
        self,
        notes,
        instrument="violin",
        behavior=None,
        traj_gain=1.0,
        noise_gain=0.6,
        noise_gate_db=-1.0,
        vibrato_source="retrieval",
        randomness=0.3,
        timing=0.0,
        continuity=0.0,
        contiguity=0.0,
        motion=1.0,
        postfx=None,
        tail_s=0.5,
    ):
        """[{midi,start,len,vel,art}] → (float32 audio, sr)。空输入返回 0.1s 静音。

        instrument: 音色(哪个 DDSP 模型发声)。
        behavior:   行为(从哪个乐器的轨迹库检索微观动态 + 用谁的包络);
                    缺省 = 同 instrument。跨乐器迁移 = 两者不同,
                    如 behavior="guitar", instrument="violin" → 小提琴音色按吉他行为演奏。
                    也可直接传一个带 .predict 的行为模型对象(服务端 captured 库)。
        traj_gain:  轨迹强度(1.0 = 原样;越大微观动态越夸张,音准偏差也随之放大)。
        noise_gain / noise_gate_db: DDSP 噪声分支音量与响度门限(压底噪)。
        vibrato_source: "retrieval"(默认) / "model"(用学习的颤音模型合成 pitch vibrato)。
        randomness: 颤音随机性 0–1(仅 vibrato_source="model" 时生效)。
        timing: 0(默认,关) / 0–1 强度:渲染前对音符表施加从行为源检索的
                expressive timing(articulation 缩短 + agogic onset 微偏移)。
        continuity: 拼接代价重排序的 concat_lambda(0=关)。
        contiguity: 连续检索奖励(0=off):优先取上一音在源乐句里的下一个音。
        motion: L0 音色吃行为库第 3/4 维(亮度/气声运动)的强度,0=关;DDSP 忽略。
        postfx: {"brightness": 0–1, "reverb": 0–1} 后处理(见 neural/postfx.py)。
        tail_s: 末音之后保留的尾音长度;网页端写回 DAW 用 0.05 保时间轴严格同长。
        """
        import torch

        from neural.perform import control_from_behavior
        from neural.postfx import apply_reverb, apply_tilt

        if not notes:
            return np.zeros(int(0.1 * self.sr), np.float32), self.sr
        notes = self._monophonize(notes)
        # 音色解码器:内置名 → DDSP;带 render_control 的对象(runtime.engine.Engine,
        # 取色 L0 音色)→ 同一 (f0, loudness) 接口的加法合成。两者共用下面全部行为逻辑。
        l0 = hasattr(instrument, "render_control")
        v_timbre = None if l0 else self.voice(instrument)
        sr = int(instrument.sr) if l0 else self.sr

        behav_nn, behav_envelope = self.resolve_behavior(
            behavior, timbre_voice=v_timbre, instrument_name=None if l0 else instrument
        )

        if timing and timing > 0:
            notes = apply_timing(notes, behav_nn, strength=float(timing))

        vib_pred = None
        inst_idx = VIBRATO_INST_IDX.get(instrument, 0)
        if vibrato_source == "model":
            vib_pred = self._get_vibrato_predictor()

        rendered = []
        for ph in self._phrases(notes):
            score = [{"midi": p["midi"], "dur": p["dur"], "art": p["art"]} for p in ph]
            ctl = control_from_behavior(
                score,
                behav_nn,
                pitch_glide=False,
                envelope=behav_envelope,
                traj_gain=traj_gain,
                vibrato_source=vibrato_source,
                vibrato_predictor=vib_pred,
                instrument_idx=inst_idx,
                randomness=randomness,
                concat_lambda=float(continuity),
                contiguity=float(contiguity),
                extra_dims=l0,
            )
            f0, loud = ctl[0], ctl[1]
            if l0:
                # 乐句内音符起点(控制流坐标):逐音时长累加,减去 crossfade 重叠
                cf_s = 0.02
                ons, acc = [], 0.0
                for k, p in enumerate(ph):
                    ons.append(max(0.0, acc - k * cf_s))
                    acc += p["dur"]
                # 0.1 s 尾 = 80 ms 释放,乐句间隙 ≥80 ms 放得下
                y = instrument.render_control(
                    f0.numpy()[0, :, 0],
                    loud.numpy()[0, :, 0],
                    onsets_s=ons,
                    tail_s=0.1,
                    bright=ctl[2],
                    noise=ctl[3],
                    motion=float(motion),
                )
            else:
                with torch.no_grad():
                    y = (
                        v_timbre.model(
                            f0.to(self.device),
                            loud.to(self.device),
                            noise_gain=noise_gain,
                            noise_gate_db=noise_gate_db,
                        )
                        .cpu()
                        .numpy()[0]
                    )
            y = y.astype(np.float64) * min(1.0, np.mean([p["vel"] for p in ph]) / 110.0)
            fade = max(1, int(EDGE_FADE_S * sr))
            y[:fade] *= np.linspace(0, 1, fade)
            y[-fade:] *= np.linspace(1, 0, fade)
            rendered.append((ph[0]["t0"], y))

        end = max(t0 + len(y) / sr for t0, y in rendered) + float(tail_s)
        out = np.zeros(int(np.ceil(end * sr)), np.float64)
        for t0, y in rendered:
            i0 = int(round(t0 * sr))
            out[i0 : i0 + len(y)] += y

        fx = postfx or {}
        out = apply_tilt(out, sr, fx.get("brightness", 0.5))
        out = apply_reverb(out, sr, fx.get("reverb", 0.0))
        peak = np.max(np.abs(out)) + 1e-9
        return (out / peak * 0.9).astype(np.float32), sr

    def _get_vibrato_predictor(self):
        if not hasattr(self, "_vib_pred"):
            from neural.vibrato_model import VibratoPredictor

            ckpt = CKPT_DIR / "vibrato_model.pt"
            self._vib_pred = VibratoPredictor(ckpt, device="cpu")
        return self._vib_pred


_instance = None


def get_neural_take():
    """进程级单例(bridge 懒加载用)。"""
    global _instance
    if _instance is None:
        _instance = NeuralTake()
    return _instance
