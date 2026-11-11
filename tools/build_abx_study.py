"""Generate the cross-instrument style identity ABX listening page (audio embedded as base64).

Design: each question gives a Reference (performer A, instrument X),
      then two Candidates (both with the instrument Y timbre):
        - one uses A's style library
        - one uses B's style library
      Question: which candidate's playing style is closer to the reference?

If listeners significantly choose the A-style, style remains identifiable across instruments.

Run: cd ~/Desktop/MT/project/quill && conda run -n mt python tools/build_abx_study.py
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
        "The Reference is a violin performance. Both Candidates have a trumpet timbre. Whose playing style is closer to the Reference?",
    ),
    # Q2: same but folk melody
    (
        "Q2",
        EV / "x_folk_viB_viT.wav",
        EV / "x_folk_viB_trT.wav",
        EV / "x_folk_trB_trT.wav",
        "The Reference is a violin performance. Both Candidates have a trumpet timbre. Whose playing style is closer to the Reference?",
    ),
    # Q3: trumpet behavior — can you recognize it on violin?
    (
        "Q3",
        EV / "x_lyrical_trB_trT.wav",  # ref: trumpet behavior × trumpet timbre
        EV / "x_lyrical_trB_viT.wav",  # match: trumpet behavior × violin timbre
        EV / "x_lyrical_viB_viT.wav",  # foil: violin behavior × violin timbre
        "The Reference is a trumpet performance. Both Candidates have a violin timbre. Whose playing style is closer to the Reference?",
    ),
    # Q4: folk melody, trumpet → violin
    (
        "Q4",
        EV / "x_folk_trB_trT.wav",
        EV / "x_folk_trB_viT.wav",
        EV / "x_folk_viB_viT.wav",
        "The Reference is a trumpet performance. Both Candidates have a violin timbre. Whose playing style is closer to the Reference?",
    ),
    # Q5: guitar behavior — recognize on violin?
    (
        "Q5",
        EV / "x_lyrical_guB_guT.wav",  # ref: guitar behavior × guitar timbre
        EV / "x_lyrical_guB_viT.wav",  # match: guitar behavior × violin timbre
        EV / "x_lyrical_viB_viT.wav",  # foil: violin behavior × violin timbre
        "The Reference is a guitar performance. Both Candidates have a violin timbre. Whose playing style is closer to the Reference?",
    ),
    # Q6: violin behavior — recognize on guitar?
    (
        "Q6",
        EV / "x_folk_viB_viT.wav",
        EV / "x_folk_viB_guT.wav",
        EV / "x_folk_guB_guT.wav",
        "The Reference is a violin performance. Both Candidates have a guitar timbre. Whose playing style is closer to the Reference?",
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
            key[qid] = "A=match(correct) B=foil"
        else:
            a64, b64_ = f64, m64
            key[qid] = "A=foil B=match(correct)"

        qs_html.append(f'''
    <div class="q">
      <h3>{qid}. {question}</h3>
      <div class="players">
        <div><strong>Reference:</strong> <audio controls preload="none" src="data:audio/wav;base64,{ref64}"></audio></div>
        <div><strong>Candidate A:</strong> <audio controls preload="none" src="data:audio/wav;base64,{a64}"></audio></div>
        <div><strong>Candidate B:</strong> <audio controls preload="none" src="data:audio/wav;base64,{b64_}"></audio></div>
      </div>
      <div class="opts">
        <label><input type="radio" name="{qid}" value="A"> A is closer to the Reference</label>
        <label><input type="radio" name="{qid}" value="B"> B is closer to the Reference</label>
        <label><input type="radio" name="{qid}" value="unsure"> Not sure</label>
      </div>
    </div>''')

    names_js = ",".join(f'"{t[0]}"' for t in TRIALS)

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Quill Listening Test II — Playing Style Identification (about 10 minutes)</title>
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
<h2>Quill Listening Test II — Playing Style Identification</h2>
<div class="info">
<p><strong>Please wear headphones.</strong> Each question has three audio clips:</p>
<ul>
  <li><strong>Reference</strong>: a performer playing a certain instrument</li>
  <li><strong>Candidate A / B</strong>: two renderings with the timbre of a <em>different instrument</em></li>
</ul>
<p>Please judge: which Candidate's <strong>playing style</strong> (vibrato, dynamics, sense of rhythm) is closer to the Reference?<br>
Ignore the timbre difference and focus only on the way of playing. You can replay as often as you like. About 10 minutes.</p>
</div>
{"".join(qs_html)}
<h3>When finished, click here and send the generated text back to me:</h3>
<button onclick="collect()">Generate answers</button>
<textarea id="result" readonly placeholder="Answers appear here; select all, copy and send back"></textarea>
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
    print("\n=== Answer KEY (do not send to participants) ===")
    for k, v in key.items():
        print(f"  {k}: {v}")
    print("\n6 questions, each with three options (A/B/unsure)")
    print("chance level = 50% (after excluding unsure)")
    print("N>=13, binomial test p<0.05 needs >=10/13 correct")


if __name__ == "__main__":
    main()
