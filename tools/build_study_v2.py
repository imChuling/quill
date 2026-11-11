"""Listening test v2 question builder: a single HTML page, per-participant randomisation + machine-readable answer codes.

Design in eval/listening/prereg_v2.md (preregistered before collection).
- Participants enter a two-digit ID -> the front end seeds a PRNG -> independent question order and A/B positions per participant.
- The answer code records the **stimulus index** chosen for each question (match/foil is not exposed),
  decoded with tools/decode_study_v2.py + the manifest.

Run: cd ~/Desktop/MT/project/quill && conda run -n mt python tools/build_study_v2.py
Outputs: checkpoints/study_v2.html (sent to participants)
      eval/listening/study_v2_manifest.json (decoding key, do not send)
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

# ---- Stimulus list (the index is the id in the answer code) ----
STIMULI = [
    EV / "lyrical_v0.wav",  # 0
    EV / "lyrical_flat.wav",  # 1
    EV / "folk_v0.wav",  # 2
    EV / "folk_allplain.wav",  # 3
    EV / "lyrical_v2.wav",  # 4
    EV / "x_folk_viB_viT.wav",  # 5
    EV / "x_folk_guB_viT.wav",  # 6
    REF / "ddsp_inference.wav",  # 7  (MIDI-DDSP baseline)
    EV / "g22_lyrical_viB_glT.wav",  # 8
    EV / "g22_lyrical_glB_glT.wav",  # 9
    EV / "x_lyrical_viB_viT.wav",  # 10
    EV / "x_lyrical_viB_trT.wav",  # 11
    EV / "x_lyrical_trB_trT.wav",  # 12
    EV / "x_folk_viB_viT.wav",  # 13 (=5, but a separate index for easier decoding)
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

# main listening test: (qid, candidate idxA, candidate idxB, question, type, for recording the hypothesised direction)
MAIN = [
    ("M1", 0, 1, "Which clip sounds more like a human performance?", "ab"),
    ("M2", 2, 3, "Which clip has richer expressive variation (vibrato/slides)?", "ab"),
    ("M3", 4, 0, "Which clip is more natural?", "ab"),
    ("M4", 5, 6, "Both clips have the same instrument timbre. Does the way of playing (vibrato, slides, dynamic habits) sound the same?", "same"),
    ("M5", 0, 7, "Rate the overall naturalness of each clip (the melodies differ; do not compare the melodies themselves)", "rate"),
    ("M6", 8, 9, "Both clips have a glass timbre. Does the way of playing sound the same?", "same"),
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
    "The Reference is a performance. The two Candidates have the same instrument timbre. "
    "Which Candidate's <b>pitch and dynamic contour (way of playing)</b> is closer to the Reference?"
)


def b64(p):
    return base64.b64encode(Path(p).read_bytes()).decode()


def main():
    missing = [str(p) for p in STIMULI if not p.exists()]
    if missing:
        print("missing stimulus files:", *missing, sep="\n  ")
        sys.exit(1)

    audio_js = json.dumps([b64(p) for p in STIMULI])
    main_js = json.dumps(MAIN, ensure_ascii=False)
    abx_js = json.dumps(ABX)
    abx_q = ABX_QUESTION

    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>Quill Listening Test v2</title>
<style>
body{{font-family:sans-serif;max-width:760px;margin:2em auto;padding:0 1em;line-height:1.6}}
.card{{border:1px solid #ccc;border-radius:8px;padding:1em;margin:1.2em 0}}
audio{{width:100%;margin:.3em 0}}
button{{padding:.4em 1em;margin:.2em}}
#code{{font-family:monospace;background:#f5f5f5;padding:1em;word-break:break-all}}
.hide{{display:none}}
label{{margin-right:1em}}
</style></head><body>
<h2>Quill Listening Test v2 (about 12 minutes)</h2>
<div id="intro" class="card">
<p>Please wear headphones. First enter a <b>self-chosen two-digit ID</b> (e.g. 47); it sets your question order and anonymously identifies your answers.</p>
<input id="pid" type="text" maxlength="2" placeholder="two digits" style="font-size:1.2em;width:6em">
<button onclick="start()">Start</button>
</div>
<div id="quiz" class="hide"></div>
<div id="done" class="hide card">
<p><b>Done!</b> Please copy the whole answer code below and send it back to the organiser:</p>
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
  if(!/^\\d{{2}}$/.test(pid)){{alert('Please enter two digits');return}}
  const rng = mulberry32(parseInt(pid,10)+20260809);
  const mainOrder = shuffled(MAIN,rng);
  const abxOrder = shuffled(ABX,rng);
  let h = '';
  let qn = 1;
  for(const [qid,ia,ib,q,typ] of mainOrder){{
    const flip = rng()<0.5;
    const [pA,pB] = flip? [ib,ia] : [ia,ib];
    h += `<div class="card"><b>Question ${{qn++}}</b><p>${{q}}</p>
      <div>A: ${{au(pA)}}</div><div>B: ${{au(pB)}}</div>`;
    if(typ==='ab'||typ==='same'){{
      const opts = typ==='ab'? [['A',pA],['B',pB]] : [['Same','same'],['Different','diff']];
      h += opts.map(([lab,val])=>
        `<label><input type="radio" name="${{qid}}" value="${{val}}">${{lab}}</label>`).join('');
    }} else {{
      h += ['A','B'].map((lab,k)=>{{
        const idx = k===0? pA:pB;
        return `<div>${{lab}} rating: `+[1,2,3,4,5].map(v=>
          `<label><input type="radio" name="${{qid}}_${{idx}}" value="${{v}}">${{v}}</label>`).join('')+`</div>`;
      }}).join('');
    }}
    h += `</div>`;
  }}
  for(const [qid,ref,m,f] of abxOrder){{
    const flip = rng()<0.5;
    const [pA,pB] = flip? [f,m] : [m,f];
    h += `<div class="card"><b>Question ${{qn++}}</b><p>${{ABX_Q}}</p>
      <div>Reference: ${{au(ref)}}</div>
      <div>Candidate A: ${{au(pA)}}</div><div>Candidate B: ${{au(pB)}}</div>
      <label><input type="radio" name="${{qid}}" value="${{pA}}">A</label>
      <label><input type="radio" name="${{qid}}" value="${{pB}}">B</label>
      <label><input type="radio" name="${{qid}}" value="u">Not sure</label></div>`;
  }}
  h += `<button onclick="finish()">Submit</button>`;
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
  if(incomplete>0 && !confirm(`${{incomplete}} questions unanswered. Submit anyway?`)) return;
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
    print(f"manifest -> {OUT_MANIFEST} (decoding key, do not send to participants)")


if __name__ == "__main__":
    main()
