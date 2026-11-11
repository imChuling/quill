# Listening test v2 preregistration (finalised before data collection)

Finalisation date: 2026-08-09 (before any v2 answers were collected).
This document is not modified once data collection starts; the analysis follows it, and any deviation must be disclosed in the paper.

## Relation to v1 (2026-08-05 / N=13, N=18)

v1 per-participant data was not kept and there was no randomisation across participants; v2 results **replace entirely** the
v1 numbers in the paper. v1 becomes an unreported pilot. Participants may overlap with v1, which the paper discloses as
"participants overlap with an earlier pilot using similar stimuli".

## Design

- A single self-contained HTML (`checkpoints/study_v2.html`), with embedded audio.
- Participants enter a self-chosen two-digit ID -> the front end seeds a PRNG with the ID:
  **independent per participant** question order and A/B positions (counterbalancing).
- At the end the page generates an answer code (recording the ID, the shuffle mapping, and the stimulus index chosen for each question),
  which participants copy and send back; `tools/decode_study_v2.py` decodes it into an anonymous CSV.

## Questions

### Part 1 main listening test (6 questions, forced choice / rating)

| # | Comparison | Hypothesised direction |
|---|---|---|
| M1 | v0 retrieved behaviour vs flat, no behaviour (lyrical) | choose v0 |
| M2 | with articulation labels vs all plain (folk) | choose labelled |
| M3 | v2 parametric vs v0 retrieval (lyrical) | no direction (exploratory) |
| M4 | violin behaviour vs guitar behaviour, same violin timbre (folk) | hear a difference |
| M5 | naturalness rating: Quill vs MIDI-DDSP (1-5 each) | no direction (baseline reference) |
| M6 | glass timbre × violin behaviour vs glass timbre × glass behaviour (lyrical, +12) | hear a difference |

### Part 2 cross-instrument ABX (8 questions)

Question wording (v2 revision): **"Which Candidate's pitch and dynamic contour (way of playing)
is closer to the Reference?"** Options A / B / unsure.

| # | Reference | Candidates (same timbre) | match |
|---|---|---|---|
| X1 | viB×viT lyrical | trT:viB vs trB | viB |
| X2 | viB×viT folk | trT:viB vs trB | viB |
| X3 | trB×trT lyrical | viT:trB vs viB | trB |
| X4 | trB×trT folk | viT:trB vs viB | trB |
| X5 | guB×guT lyrical | viT:guB vs viB | guB |
| X6 | guB×guT folk | viT:guB vs viB | guB |
| X7 | viB×viT lyrical(+12) | glassT:viB vs glB | viB |
| X8 | glB×glT folk(+12) | viT:glB vs viB | glB |

X7/X8 are new for glass. X8 is a perceptual test of the behaviour-bandwidth hypothesis: expected to be
**near chance** (glass behaviour transferred to violin is perceptually weak); reported whatever the result.

## Statistical plan

- Main listening test binary questions (M1/M2/M4/M6): one-sided binomial test (direction as in the table above), M3 two-sided.
- M5: paired Wilcoxon (ratings).
- ABX: unsure **excluded from accuracy but the count is reported** (preregistered rule);
  pooled one-sided binomial; per-trial binomial + Bonferroni (0.05/8);
  per-participant accuracy + Wilcoxon signed-rank vs 0.5.
  X7/X8 reported separately, not combined with X1–X6 into the "cross-instrument identifiability" claim.
- Sample target: aim for ≥18, accept the number actually collected; no sequential peeking followed by additional collection.

## Exclusion rules

- Multiple submissions with the same ID: the first is kept.
- Incomplete/undecodable answer code: the participant is excluded entirely and the count is reported.
- No other post-hoc exclusions.
