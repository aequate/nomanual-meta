# Machine change report: filming_05

## Change after experiment 12

- Alarm: cause **collapse**, statistic 3.8 against the threshold 10.0.
- Before: A moves the tip right, only while B held; D moves the tip down, only while B held; C toggles mode; above y = 0.30, using A makes the tip jump right once.
- After: A moves the tip left, only while C held; B moves the tip right; D toggles mode (repair edits: A moves the tip left).

### What changed

| control | before | now |
|---|---|---|
| A | moves the tip right, only while B held | moves the tip left, only while C held |
| B | interlock (hold it for A, D) | moves the tip right |
| C | toggles a mode | interlock (hold it for A) |
| D | moves the tip down, only while B held | toggles a mode |

| rules removed (support before the change) | state | evidence (experiments) |
|---|---|---|
| A moves the tip right | supported | 1, 2, 3, 4, 5, 6, 7, 9 |
| D moves the tip down | supported | 1, 2, 3, 4, 5, 6, 7, 9 |
| A only works while B is held | ambiguous | – |
| D only works while B is held | supported | 1, 2, 3, 4, 5, 6, 7, 9 |
| pressing C toggles a hidden mode | supported | 1, 2, 3, 4, 5, 6, 7, 9 |
| above y = 0.30, using A makes the tip jump right once | ambiguous | – |

| rules added (state at the end of the run) | state | evidence (experiments) |
|---|---|---|
| A moves the tip left | not in the logged rule list | – |
| B moves the tip right | contradicted | 39, 46, 47, 48 |
| A only works while C is held | not in the logged rule list | – |
| B works without holding C | not in the logged rule list | – |
| pressing D toggles a hidden mode | contradicted | 39, 46, 47, 48 |

Rules kept: none.

### Evidence

**Trigger, experiment 12** (A-1 B+1 1.00s): seen (+0.051, -0.005).
- old explanation predicts (-0.012, +0.000): refuted on its own (z = 5.1 > 4.5, drop arming carried from reset).
- new explanation predicts (+0.036, +0.000): fits (z = 1.1).
- Read as the start of a new segment (drop arming unknown, as the agent treats data after an alarm): old predicts (-0.012, +0.000) at 5.1σ, refuted on its own; new predicts (+0.036, +0.000) at 1.1σ.
- Speeds: the old explanation's, fitted on the 12 pre-change experiments, for both.

**Fit-before test** (z ≤ 4.5, each with its own speeds): old fits 12 of 12, new fits 7 of 12. The old explanation fits everything before the change and the new one does not: **the machine changed.**

**After the alarm:** 3 of 26 later experiments moved the tip and separate the two explanations:

| experiment | action | seen | old z | new z |
|---|---|---|---|---|
| 19 | A+1 B-1 2.00s | (-0.075, +0.009) | 7.3 | 0.5 |
| 20 | A+1 B-1 2.00s | (-0.065, +0.001) | 6.7 | 0.4 |
| 22 | A+1 B-1 2.00s | (-0.052, +0.002) | 5.7 | 1.0 |

### Remaining uncertainty

- Rules of the new explanation not supported at the end of the run: A moves the tip left (not in the logged rule list); B moves the tip right (contradicted); A only works while C is held (not in the logged rule list); B works without holding C (not in the logged rule list); pressing D toggles a hidden mode (contradicted).
- Unknown (grey) area of the board at the end: 89%.
- Tasks after the change: reach success (after 38 experiments), park success (after 59 experiments), reach success (after 72 experiments).

*Written by `nomanual/core/change_report.py` from `events.jsonl` alone; no model wrote any of this report.*

## Change after experiment 39

- Alarm: cause **collapse**, statistic 4.0 against the threshold 10.0.
- Before: B moves the tip right, only while A held; D moves the tip down; C toggles mode; after pressing C (alternate mode), D moves at 2x speed.
- After: B moves the tip up, only while A held; D moves the tip left; C toggles mode; after pressing C (alternate mode), D moves at 2x speed (repair edits: B moves the tip up; D moves the tip left).

### What changed

| control | before | now |
|---|---|---|
| A | interlock (hold it for B) | interlock (hold it for B) (same) |
| B | moves the tip right, only while A held | moves the tip up, only while A held |
| C | toggles a mode | toggles a mode (same) |
| D | moves the tip down | moves the tip left |

| rules removed (support before the change) | state | evidence (experiments) |
|---|---|---|
| B moves the tip right | supported | 11, 12, 13, 15, 18, 19, 20, 21 |
| D moves the tip down | supported | 11, 12, 13, 15, 18, 19, 20, 21 |

| rules added (state at the end of the run) | state | evidence (experiments) |
|---|---|---|
| B moves the tip up | supported | 38, 39, 41, 42, 44, 45, 46, 47 |
| D moves the tip left | supported | 38, 39, 41, 42, 44, 45, 46, 47 |

Rules kept: B only works while A is held; D works without holding A; pressing C toggles a hidden mode; after pressing C (alternate mode), D moves at 2x speed.

### Evidence

**Trigger, experiment 39** (A+1 B-1 2.00s): seen (+0.004, -0.066).
- old explanation predicts (-0.056, +0.000): refuted on its own (z = 5.2 > 4.5, drop arming carried from reset).
- new explanation predicts (+0.000, -0.056): fits (z = 0.7).
- Read as the start of a new segment (drop arming unknown, as the agent treats data after an alarm): old predicts (-0.056, +0.000) at 5.2σ, refuted on its own; new predicts (+0.000, -0.056) at 0.7σ.
- Speeds: the old explanation's, fitted on the 27 pre-change experiments, for both.

**Fit-before test** (z ≤ 4.5, each with its own speeds): old fits 27 of 27, new fits 13 of 27. The old explanation fits everything before the change and the new one does not: **the machine changed.**

**After the alarm:** 31 of 32 later experiments moved the tip, but none separates the two explanations by 4σ.

### Remaining uncertainty

- Rules of the new explanation not supported at the end of the run: after pressing C (alternate mode), D moves at 2x speed (ambiguous).
- Unknown (grey) area of the board at the end: 89%.
- Tasks after the change: park success (after 59 experiments), reach success (after 72 experiments).

*Written by `nomanual/core/change_report.py` from `events.jsonl` alone; no model wrote any of this report.*
