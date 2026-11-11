"""听测 v2 出题:单页 HTML,per-participant 随机化 + 机读答案码。

设计见 eval/listening/prereg_v2.md(先预注册后收集)。
- 参与者输入两位数字编号 → 前端种子化 PRNG → 逐人独立题序与 A/B 位置。
- 答案码记录逐题所选的**刺激索引**(不暴露 match/foil),
  用 tools/decode_study_v2.py + manifest 解码。

跑: cd ~/Desktop/MT/project/quill && conda run -n mt python tools/build_study_v2.py
产出: checkpoints/study_v2.html(发参与者)
      eval/listening/study_v2_manifest.json(解码 key,勿发)
"""

import base64
import json
import sys
from pathlib import Path

QUILL = Path(__file__).resolve().parent.parent
EV = QUILL / "checkpoints" / "eval"
REF = QUILL / "checkpoints" / "midi_ddsp_ref"
OUT_HTML = QUILL / "checkpoints" / "study_v2.html"
OUT_MANIFEST = QUILL / "eval" / "listening" / "study_v2_manifest.json"

# ---- 刺激清单(索引即答案码里的 id) ----
STIMULI = [
    EV / "lyrical_v0.wav",  # 0
    EV / "lyrical_flat.wav",  # 1
    EV / "folk_v0.wav",  # 2
    EV / "folk_allplain.wav",  # 3
    EV / "lyrical_v2.wav",  # 4
    EV / "x_folk_viB_viT.wav",  # 5
    EV / "x_folk_guB_viT.wav",  # 6
    REF / "ddsp_inference.wav",  # 7  (MIDI-DDSP 基准)
    EV / "g22_lyrical_viB_glT.wav",  # 8
    EV / "g22_lyrical_glB_glT.wav",  # 9
    EV / "x_lyrical_viB_viT.wav",  # 10
    EV / "x_lyrical_viB_trT.wav",  # 11
    EV / "x_lyrical_trB_trT.wav",  # 12
    EV / "x_folk_viB_viT.wav",  # 13 (=5, 但独立索引便于解码)
    EV / "x_folk_viB_trT.wav",  # 14
    EV / "x_folk_trB_trT.wav",  # 15
    EV / "x_lyrical_trB_viT.wav",  # 16
    EV / "x_folk_trB_viT.wav",  # 17
    EV / "x_lyrical_guB_guT.wav",  # 18
    EV / "x_lyrical_guB_viT.wav",  # 19
    EV / "x_folk_guB_guT.wav",  # 20
    EV / "x_folk_guB_viT.wav",  # 21
    EV / "g22_lyrical_viB_viT.wav",  # 22
    EV / "g22_folk_glB_glT.wav",  # 23
    EV / "g22_folk_glB_viT.wav",  # 24
    EV / "g22_folk_viB_viT.wav",  # 25
]

# 主听测: (qid, idxA候选, idxB候选, 问题, 类型, 方向假设记录用)
MAIN = [
    ("M1", 0, 1, "哪段听起来更像真人演奏?", "ab"),
    ("M2", 2, 3, "哪段的表现力变化(揉弦/滑音)更丰富?", "ab"),
    ("M3", 4, 0, "哪段更自然?", "ab"),
    ("M4", 5, 6, "两段乐器音色相同。演奏方式(揉弦、滑音、力度习惯)听起来一样吗?", "same"),
    ("M5", 0, 7, "分别给两段的整体自然度打分(旋律不同,不比较旋律本身)", "rate"),
    ("M6", 8, 9, "两段都是玻璃杯音色。演奏方式听起来一样吗?", "same"),
]

# ABX: (qid, ref, match, foil)
ABX = [
    ("X1", 10, 11, 12),
    ("X2", 13, 14, 15),
    ("X3", 12, 16, 10),
    ("X4", 15, 17, 13),
    ("X5", 18, 19, 10),
    ("X6", 20, 21, 13),
    ("X7", 22, 8, 9),
    ("X8", 23, 24, 25),
]

ABX_QUESTION = (
    "Reference 是一段演奏。两个 Candidate 是同一种乐器音色。"
    "哪个 Candidate 的<b>音高与力度起伏(演奏方式)</b>更接近 Reference?"
)


def b64(p):
    return base64.b64encode(Path(p).read_bytes()).decode()


def main():
    missing = [str(p) for p in STIMULI if not p.exists()]
    if missing:
        print("缺刺激文件:", *missing, sep="\n  ")
        sys.exit(1)

    audio_js = json.dumps([b64(p) for p in STIMULI])
    main_js = json.dumps(MAIN, ensure_ascii=False)
    abx_js = json.dumps(ABX)
    abx_q = ABX_QUESTION

    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>Quill 听测 v2</title>
<style>
body{{font-family:sans-serif;max-width:760px;margin:2em auto;padding:0 1em;line-height:1.6}}
.card{{border:1px solid #ccc;border-radius:8px;padding:1em;margin:1.2em 0}}
audio{{width:100%;margin:.3em 0}}
button{{padding:.4em 1em;margin:.2em}}
#code{{font-family:monospace;background:#f5f5f5;padding:1em;word-break:break-all}}
.hide{{display:none}}
label{{margin-right:1em}}
</style></head><body>
<h2>Quill 听测 v2(约 12 分钟)</h2>
<div id="intro" class="card">
<p>请戴耳机。先输入一个<b>自选两位数字编号</b>(例如 47),它决定你的题目顺序,并用于匿名标识你的回答。</p>
<input id="pid" type="text" maxlength="2" placeholder="两位数字" style="font-size:1.2em;width:6em">
<button onclick="start()">开始</button>
</div>
<div id="quiz" class="hide"></div>
<div id="done" class="hide card">
<p><b>完成!</b>请复制下面整段答案码发回给组织者:</p>
<div id="code"></div>
</div>
<script>
const AUDIO = {audio_js};
const MAIN = {main_js};
const ABX = {abx_js};
const ABX_Q = `{abx_q}`;
let answers = {{}}, pid = "";

function mulberry32(a){{return function(){{a|=0;a=a+0x6D2B79F5|0;
let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;
return((t^t>>>14)>>>0)/4294967296}}}}
function shuffled(arr,rng){{const a=arr.slice();
for(let i=a.length-1;i>0;i--){{const j=Math.floor(rng()*(i+1));[a[i],a[j]]=[a[j],a[i]]}}
return a}}
function au(i){{return `<audio controls preload="none" src="data:audio/wav;base64,${{AUDIO[i]}}"></audio>`}}

function start(){{
  pid = document.getElementById('pid').value.trim();
  if(!/^\\d{{2}}$/.test(pid)){{alert('请输入两位数字');return}}
  const rng = mulberry32(parseInt(pid,10)+20260809);
  const mainOrder = shuffled(MAIN,rng);
  const abxOrder = shuffled(ABX,rng);
  let h = '';
  let qn = 1;
  for(const [qid,ia,ib,q,typ] of mainOrder){{
    const flip = rng()<0.5;
    const [pA,pB] = flip? [ib,ia] : [ia,ib];
    h += `<div class="card"><b>第 ${{qn++}} 题</b><p>${{q}}</p>
      <div>A: ${{au(pA)}}</div><div>B: ${{au(pB)}}</div>`;
    if(typ==='ab'||typ==='same'){{
      const opts = typ==='ab'? [['A',pA],['B',pB]] : [['一样','same'],['不一样','diff']];
      h += opts.map(([lab,val])=>
        `<label><input type="radio" name="${{qid}}" value="${{val}}">${{lab}}</label>`).join('');
    }} else {{
      h += ['A','B'].map((lab,k)=>{{
        const idx = k===0? pA:pB;
        return `<div>${{lab}} 评分: `+[1,2,3,4,5].map(v=>
          `<label><input type="radio" name="${{qid}}_${{idx}}" value="${{v}}">${{v}}</label>`).join('')+`</div>`;
      }}).join('');
    }}
    h += `</div>`;
  }}
  for(const [qid,ref,m,f] of abxOrder){{
    const flip = rng()<0.5;
    const [pA,pB] = flip? [f,m] : [m,f];
    h += `<div class="card"><b>第 ${{qn++}} 题</b><p>${{ABX_Q}}</p>
      <div>Reference: ${{au(ref)}}</div>
      <div>Candidate A: ${{au(pA)}}</div><div>Candidate B: ${{au(pB)}}</div>
      <label><input type="radio" name="${{qid}}" value="${{pA}}">A</label>
      <label><input type="radio" name="${{qid}}" value="${{pB}}">B</label>
      <label><input type="radio" name="${{qid}}" value="u">不确定</label></div>`;
  }}
  h += `<button onclick="finish()">提交</button>`;
  document.getElementById('intro').classList.add('hide');
  const qz = document.getElementById('quiz');
  qz.innerHTML = h; qz.classList.remove('hide');
}}

function finish(){{
  const parts = [`QV2`,`id=${{pid}}`];
  let incomplete = 0;
  for(const [qid,ia,ib,q,typ] of MAIN){{
    if(typ==='rate'){{
      for(const idx of [ia,ib]){{
        const r = document.querySelector(`input[name="${{qid}}_${{idx}}"]:checked`);
        if(r) parts.push(`${{qid}}r${{idx}}=${{r.value}}`); else incomplete++;
      }}
    }} else {{
      const r = document.querySelector(`input[name="${{qid}}"]:checked`);
      if(r) parts.push(`${{qid}}=${{r.value}}`); else incomplete++;
    }}
  }}
  for(const [qid] of ABX){{
    const r = document.querySelector(`input[name="${{qid}}"]:checked`);
    if(r) parts.push(`${{qid}}=${{r.value}}`); else incomplete++;
  }}
  if(incomplete>0 && !confirm(`还有 ${{incomplete}} 题未答,确认提交?`)) return;
  document.getElementById('code').textContent = parts.join('|');
  document.getElementById('done').classList.remove('hide');
  window.scrollTo(0,document.body.scrollHeight);
}}
</script></body></html>"""

    OUT_HTML.write_text(html)
    manifest = {
        "stimuli": {str(i): str(p.relative_to(QUILL)) for i, p in enumerate(STIMULI)},
        "main": [{"qid": q[0], "idxA": q[1], "idxB": q[2], "type": q[4]} for q in MAIN],
        "abx": [{"qid": q[0], "ref": q[1], "match": q[2], "foil": q[3]} for q in ABX],
    }
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    size_mb = OUT_HTML.stat().st_size / 1e6
    print(f"study_v2.html ({size_mb:.1f} MB) -> {OUT_HTML}")
    print(f"manifest -> {OUT_MANIFEST}(解码 key,勿发参与者)")


if __name__ == "__main__":
    main()
