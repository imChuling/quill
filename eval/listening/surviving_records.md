# Surviving listening test records and data retention statement

## Conclusion

**Per-participant raw answers from both listening tests were not kept.** The only auditable data are the
summary counts in the project log at the time. Accordingly, the paper (2026-08 revision) keeps only statistics supported by the surviving summaries:

- Kept: main listening test Q2 11/13 (one-sided binomial p=0.011), Q4 13/13 (p<0.001);
  ABX pooled 98/107 (one-sided binomial p<1e-5).
- Removed: the ABX per-trial Bonferroni claim and the per-listener 91.7% Wilcoxon claim,
  since the per-trial/per-participant data they need does not exist and cannot be audited.

## Surviving records and sources

### Main listening test (N=13, collected 2026-08-05)

Source: `docs/ismir_lbd_abstract_draft_2026-08-04.md` (the project log at the time, per-question summary):

| Question | Result |
|---|---|
| Q1 behaviour vs flat | 13/13 chose behaviour |
| Q2 labelled vs allplain | 11/13 chose labelled |
| Q3 v2 parametric vs v0 retrieval | 8/13 (tie) |
| Q4 cross-instrument behaviour perception | 13/13 heard a difference |
| Q5 naturalness | Quill 3.9/5, MIDI-DDSP 4.5/5 |

Participant makeup: 8 of 13 had formal musical training (verbal summary at the time, no per-participant record).

### ABX (N=18, 6 trials per participant)

Source: `docs/changelog.md` (summary): 108 answers in total, 1 unsure excluded,
98/107 correct (91.6%). Per-trial and per-participant details were not kept.

## Lessons and follow-up

- The `analyze_listening.py` + CSV schema in this directory are prepared for **future reruns**:
  form collection, per-participant export, per-participant randomisation, a preregistered unsure rule.
- If an N≥30 rerun takes place, its results will replace the old numbers above entirely.
