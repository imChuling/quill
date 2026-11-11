"""生成 cross-instrument style identity ABX 听测页(音频 base64 内嵌)。

设计: 每题给一个 Reference(performer A, instrument X),
      再给两个 Candidate(都是 instrument Y 音色):
        - 一个用 A 的 style library
        - 一个用 B 的 style library
      问: 哪个 candidate 的演奏风格更像 reference?

如果听众显著选择 A-style, 说明 style 跨乐器后仍可识别。

跑: cd ~/Desktop/MT/project/quill && conda run -n mt python tools/build_abx_study.py
"""

import base64
import random
import sys
from pathlib import Path

QUILL = Path(__file__).resolve().parent.parent
EV = QUILL / "checkpoints" / "eval"
OUT = QUILL / "checkpoints" / "abx_study.html"

random.seed(2026)

# ABX trials: (qid, reference, candidate_match, candidate_foil, description)
# reference = performer A on instrument X
# candidate_match = A-style on instrument Y (correct answer)
# candidate_foil = B-style on instrument Y
TRIALS = [
    # Q1: violin behavior is the "style" — can you recognize it on trumpet?
    (
        "Q1",
        EV / "x_lyrical_viB_viT.wav",  # ref: violin behavior × violin timbre
        EV / "x_lyrical_viB_trT.wav",  # match: violin behavior × trumpet timbre
        EV / "x_lyrical_trB_trT.wav",  # foil: trumpet behavior × trumpet timbre
        "Reference 是小提琴演奏。两个 Candidate 都是小号音色。哪个的演奏风格更像 Reference?",
    ),
    # Q2: same but folk melody
    (
        "Q2",
        EV / "x_folk_viB_viT.wav",
        EV / "x_folk_viB_trT.wav",
        EV / "x_folk_trB_trT.wav",
        "Reference 是小提琴演奏。两个 Candidate 都是小号音色。哪个的演奏风格更像 Reference?",
    ),
    # Q3: trumpet behavior — can you recognize it on violin?
    (
        "Q3",
        EV / "x_lyrical_trB_trT.wav",  # ref: trumpet behavior × trumpet timbre
        EV / "x_lyrical_trB_viT.wav",  # match: trumpet behavior × violin timbre
        EV / "x_lyrical_viB_viT.wav",  # foil: violin behavior × violin timbre
        "Reference 是小号演奏。两个 Candidate 都是小提琴音色。哪个的演奏风格更像 Reference?",
    ),
    # Q4: folk melody, trumpet → violin
    (
        "Q4",
        EV / "x_folk_trB_trT.wav",
        EV / "x_folk_trB_viT.wav",
        EV / "x_folk_viB_viT.wav",
        "Reference 是小号演奏。两个 Candidate 都是小提琴音色。哪个的演奏风格更像 Reference?",
    ),
    # Q5: guitar behavior — recognize on violin?
    (
        "Q5",
        EV / "x_lyrical_guB_guT.wav",  # ref: guitar behavior × guitar timbre
        EV / "x_lyrical_guB_viT.wav",  # match: guitar behavior × violin timbre
        EV / "x_lyrical_viB_viT.wav",  # foil: violin behavior × violin timbre
        "Reference 是吉他演奏。两个 Candidate 都是小提琴音色。哪个的演奏风格更像 Reference?",
    ),
    # Q6: violin behavior — recognize on guitar?
    (
        "Q6",
        EV / "x_folk_viB_viT.wav",
        EV / "x_folk_viB_guT.wav",
        EV / "x_folk_guB_guT.wav",
        "Reference 是小提琴演奏。两个 Candidate 都是吉他音色。哪个的演奏风格更像 Reference?",
    ),
]


def b64(p):
    return base64.b64encode(Path(p).read_bytes()).decode()


def main():
    for _, ref, match, foil, _ in TRIALS:
        for f in [ref, match, foil]:
            if not f.exists():
                print(f"MISSING: {f}")
                sys.exit(1)

    qs_html = []
    key = {}
    for qid, ref_path, match_path, foil_path, question in TRIALS:
        ref64 = b64(ref_path)
        m64 = b64(match_path)
        f64 = b64(foil_path)

        # randomize A/B order
        if random.random() < 0.5:
            a64, b64_ = m64, f64
            key[qid] = "A=match(正确) B=foil"
        else:
            a64, b64_ = f64, m64
            key[qid] = "A=foil B=match(正确)"

        qs_html.append(f'''
    <div class="q">
      <h3>{qid}. {question}</h3>
      <div class="players">
        <div><strong>Reference:</strong> <audio controls preload="none" src="data:audio/wav;base64,{ref64}"></audio></div>
        <div><strong>Candidate A:</strong> <audio controls preload="none" src="data:audio/wav;base64,{a64}"></audio></div>
        <div><strong>Candidate B:</strong> <audio controls preload="none" src="data:audio/wav;base64,{b64_}"></audio></div>
      </div>
      <div class="opts">
        <label><input type="radio" name="{qid}" value="A"> A 更像 Reference</label>
        <label><input type="radio" name="{qid}" value="B"> B 更像 Reference</label>
        <label><input type="radio" name="{qid}" value="unsure"> 说不好</label>
      </div>
    </div>''')

    names_js = ",".join(f'"{t[0]}"' for t in TRIALS)

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Quill 听测 II — 演奏风格辨认(约 10 分钟)</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 720px; margin: 2em auto;
         padding: 0 1em; line-height: 1.6; }}
  .q {{ border: 1px solid #ddd; border-radius: 8px; padding: 1em; margin: 1.2em 0; }}
  .players div {{ margin: 0.4em 0; }}
  audio {{ width: 80%; vertical-align: middle; }}
  .opts label {{ margin-right: 1.2em; }}
  #result {{ width: 100%; height: 5em; margin-top: 0.5em; }}
  button {{ font-size: 1em; padding: 0.5em 1.2em; cursor: pointer; }}
  .info {{ background: #f0f4f8; border-radius: 6px; padding: 0.8em 1em; margin: 1em 0; }}
</style>
</head>
<body>
<h2>Quill 听测 II — 演奏风格辨认</h2>
<div class="info">
<p><strong>请戴耳机。</strong>每题有三段音频:</p>
<ul>
  <li><strong>Reference</strong>: 一位演奏者在某种乐器上的演奏</li>
  <li><strong>Candidate A / B</strong>: 两段<em>不同乐器</em>音色的渲染</li>
</ul>
<p>请判断: 哪个 Candidate 的<strong>演奏风格</strong>(揉弦、力度变化、节奏感觉)更像 Reference?<br>
忽略音色差异,只关注演奏方式。可重复播放。大约 10 分钟。</p>
</div>
{"".join(qs_html)}
<h3>完成后点这里,把生成的文本发回给我:</h3>
<button onclick="collect()">生成答案</button>
<textarea id="result" readonly placeholder="答案会出现在这里,全选复制发回即可"></textarea>
<script>
function collect() {{
  const names = [{names_js}];
  let out = [];
  for (const n of names) {{
    const el = document.querySelector('input[name="'+n+'"]:checked');
    out.push(n + "=" + (el ? el.value : "?"));
  }}
  document.getElementById("result").value = "QUILL-ABX " + out.join(" ");
}}
</script>
</body>
</html>"""
    OUT.write_text(html)
    print(f"saved -> {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")
    print("\n=== 答案 KEY(别发给参与者) ===")
    for k, v in key.items():
        print(f"  {k}: {v}")
    print("\n6 题, 每题三选一(A/B/unsure)")
    print("chance level = 50% (排除 unsure 后)")
    print("N>=13, binomial test p<0.05 需要 >=10/13 正确")


if __name__ == "__main__":
    main()
