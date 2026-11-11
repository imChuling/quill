"""Listening test v2 answer-code decoder: pasted answer codes -> anonymous CSV.

Usage:
  store the answer codes sent back by participants line by line in eval/listening/raw_codes_v2.txt (one line per participant),
  then run: conda run -n mt python tools/decode_study_v2.py
Outputs:
  eval/listening/responses_main_v2.csv
  eval/listening/responses_abx_v2.csv
Exclusion rules per prereg_v2.md: duplicate IDs keep the first submission; participants whose code cannot be decoded are excluded entirely and reported.
"""

import csv
import json
import sys
from pathlib import Path

QUILL = Path(__file__).resolve().parent.parent
LST = QUILL / "eval" / "listening"
MANIFEST = json.loads((LST / "study_v2_manifest.json").read_text())

ABX_KEY = {t["qid"]: t for t in MANIFEST["abx"]}
MAIN_KEY = {q["qid"]: q for q in MANIFEST["main"]}


def parse_line(line):
    parts = line.strip().split("|")
    if not parts or parts[0] != "QV2":
        return None
    d = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            d[k] = v
    if "id" not in d:
        return None
    return d


def main():
    raw = LST / "raw_codes_v2.txt"
    if not raw.exists():
        print(f"First store the answer codes line by line in {raw}")
        sys.exit(1)
    lines = [l for l in raw.read_text().splitlines() if l.strip()]
    seen, rows, bad = set(), [], 0
    for l in lines:
        d = parse_line(l)
        if d is None:
            bad += 1
            continue
        if d["id"] in seen:  # duplicate ID: keep the first submission (preregistered)
            continue
        seen.add(d["id"])
        rows.append(d)
    print(f"total {len(lines)} lines, {len(rows)} participants decoded, {bad} lines undecodable (participant excluded)")

    with open(LST / "responses_main_v2.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["participant_id", "qid", "response"])
        for d in rows:
            pid = f"P{d['id']}"
            for qid, spec in MAIN_KEY.items():
                if spec["type"] == "rate":
                    for idx in (spec["idxA"], spec["idxB"]):
                        k = f"{qid}r{idx}"
                        if k in d:
                            w.writerow([pid, k, d[k]])
                elif qid in d:
                    v = d[qid]
                    if spec["type"] == "ab":
                        v = "A_stim" if int(v) == spec["idxA"] else "B_stim"
                    w.writerow([pid, qid, v])

    with open(LST / "responses_abx_v2.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["participant_id", "trial_id", "response"])
        for d in rows:
            pid = f"P{d['id']}"
            for qid, spec in ABX_KEY.items():
                if qid not in d:
                    continue
                v = d[qid]
                if v == "u":
                    resp = "unsure"
                elif int(v) == spec["match"]:
                    resp = "match"
                else:
                    resp = "nonmatch"
                w.writerow([pid, qid, resp])
    print(f"Wrote responses_main_v2.csv / responses_abx_v2.csv -> {LST}/")


if __name__ == "__main__":
    main()
