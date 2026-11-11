"""CLI:任意方言 MIDI → Quill 统一 IR。

跑: conda run -n mt python tools/import_midi.py <file.mid> [--json out.json]

打印检测报告与转换后的 score;--json 时另存机器可读结果。
只做导入,不渲染(渲染走现有 transfer_demo / perform 管线)。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.midi_dialect import ART_NAMES, DrumFileError, import_midi  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("midi_file")
    ap.add_argument("--json", help="把 {score, report} 存为 JSON")
    ap.add_argument("--bend-range", type=float, default=None, help="弯音范围(半音),缺省 ±2")
    args = ap.parse_args()

    try:
        score, rep = import_midi(args.midi_file, bend_range_st=args.bend_range)
    except DrumFileError as e:
        print(f"拒绝: {e}")
        sys.exit(2)

    print(
        f"方言: {rep.dialect} (置信度 {rep.confidence:.2f}"
        + (f", 备选 {rep.candidates}" if rep.candidates else "")
        + ")"
    )
    print(f"音符: {rep.n_notes_in} 进 → {rep.n_notes_out} 出")
    acts = []
    if rep.n_keyswitches_stripped:
        acts.append(f"剔除 keyswitch ×{rep.n_keyswitches_stripped} (映射 {rep.ks_map_used})")
    if rep.n_channels_merged:
        acts.append(f"合并通道 ×{rep.n_channels_merged}")
    if rep.n_bend_slide or rep.n_bend_vibrato:
        acts.append(f"bend→slide ×{rep.n_bend_slide}, bend→vibrato ×{rep.n_bend_vibrato}")
    if rep.n_chord_notes_dropped or rep.n_overlaps_trimmed:
        acts.append(f"去和弦 ×{rep.n_chord_notes_dropped}, 截重叠 ×{rep.n_overlaps_trimmed}")
    if rep.n_drum_tracks_dropped:
        acts.append(f"丢鼓轨 ×{rep.n_drum_tracks_dropped}")
    if rep.cc_energy_source:
        acts.append(f"能量提示源 {rep.cc_energy_source}")
    for a in acts:
        print(f"  - {a}")
    for w in rep.warnings:
        print(f"  ! {w}")

    print("\nscore (前 12 音):")
    for n in score[:12]:
        extra = f"  cc_energy={n['cc_energy']}" if "cc_energy" in n else ""
        print(
            f"  onset={n['onset']:7.3f}  midi={n['midi']:3d}  "
            f"dur={n['dur']:.3f}  art={ART_NAMES[n['art']]}{extra}"
        )
    if len(score) > 12:
        print(f"  … 共 {len(score)} 音")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"score": score, "report": rep.__dict__}, ensure_ascii=False, indent=2)
        )
        print(f"\n已存 {args.json}")


if __name__ == "__main__":
    main()
