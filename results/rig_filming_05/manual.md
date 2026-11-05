# Operating manual

Run filming_05: 72 experiments (reach: done, park: done, reach: done). Prose: fireworks accounts/fireworks/models/muse-spark-open. Every statement cites the experiments that tested it; statements the checker could not tie to a verified rule were deleted.

## Quick start

- No verified procedure to the target yet: the experiments so far do not give a route that stays on verified ground.

## What each control does

- Control B moves the tip up. (experiments [38](#e38), [39](#e39), [41](#e41), [42](#e42))
- Control D moves the tip left. (experiments [38](#e38), [39](#e39), [41](#e41), [42](#e42))
- Control B only works while control A is held. (experiments [38](#e38), [39](#e39), [41](#e41), [42](#e42))
- Control D works without holding control A. (experiments [38](#e38), [39](#e39), [41](#e41), [42](#e42))
- Pressing control C toggles a hidden mode. (experiments [38](#e38), [39](#e39), [41](#e41), [42](#e42))

## What the panel light means

- No panel light was found to mean anything (none was verified).

## Regions where behaviour changes

- No region where a control behaves differently was verified.

## Not yet verified

- Below y = 0.20, using B makes the tip jump down once (line between 0.20 and 0.25). Not yet verified: the remaining explanations disagree about it and no experiment has decided between them yet.
- After pressing C (alternate mode), B moves at 0.5x speed. Not yet verified: the remaining explanations disagree about it and no experiment has decided between them yet.
- After pressing C (alternate mode), B moves at 2x speed. Not yet verified: the remaining explanations disagree about it and no experiment has decided between them yet.
- After pressing C (alternate mode), D drives the other axis. Not yet verified: the remaining explanations disagree about it and no experiment has decided between them yet.
- After pressing C (alternate mode), D moves at 0.25x speed. Not yet verified: the remaining explanations disagree about it and no experiment has decided between them yet.
- After pressing C (alternate mode), D moves at 0.5x speed. Not yet verified: the remaining explanations disagree about it and no experiment has decided between them yet.
- After pressing C (alternate mode), D moves at 2x speed. Not yet verified: the remaining explanations disagree about it and no experiment has decided between them yet.
- Right of x = 0.75, B moves at 0.5x speed. Not yet verified: the remaining explanations disagree about it and no experiment has decided between them yet.
- Right of x = 0.80, B stops working. Not yet verified: the remaining explanations disagree about it and no experiment has decided between them yet.
- Right of x = 0.80, using D makes the tip jump right once (line between 0.75 and 0.80). Not yet verified: the remaining explanations disagree about it and no experiment has decided between them yet.
- Right of x = 0.80, using D makes the tip jump left once (line between 0.75 and 0.80). Not yet verified: the remaining explanations disagree about it and no experiment has decided between them yet.
- Right of x = 0.80, using D makes the tip jump up once (line between 0.75 and 0.80). Not yet verified: the remaining explanations disagree about it and no experiment has decided between them yet.
- Right of x = 0.80, using D makes the tip jump down once (line between 0.75 and 0.80). Not yet verified: the remaining explanations disagree about it and no experiment has decided between them yet.
- Right of x = 0.80, D moves at 0.25x speed (line between 0.75 and 0.80). Not yet verified: the remaining explanations disagree about it and no experiment has decided between them yet.
- Right of x = 0.75, D moves at 0.5x speed (line between 0.75 and 0.80). Not yet verified: the remaining explanations disagree about it and no experiment has decided between them yet.
- Right of x = 0.80, D moves at 2x speed (line between 0.75 and 0.80). Not yet verified: the remaining explanations disagree about it and no experiment has decided between them yet.
- Right of x = 0.80, D moves the opposite way (line between 0.75 and 0.80). Not yet verified: the remaining explanations disagree about it and no experiment has decided between them yet.
- Right of x = 0.80, D stops working (line between 0.75 and 0.80). Not yet verified: the remaining explanations disagree about it and no experiment has decided between them yet.
- Below y = 0.20, using B makes the tip jump right once (line between 0.20 and 0.25). Not yet verified: the remaining explanations disagree about it and no experiment has decided between them yet.
- Below y = 0.20, using B makes the tip jump left once (line between 0.20 and 0.25). Not yet verified: the remaining explanations disagree about it and no experiment has decided between them yet.
- Below y = 0.20, using B makes the tip jump up once (line between 0.20 and 0.25). Not yet verified: the remaining explanations disagree about it and no experiment has decided between them yet.
- Below y = 0.20, B moves at 2x speed (line between 0.20 and 0.25). Not yet verified: consistent with every experiment, but no experiment has ruled out the alternatives.
- Below y = 0.20, using D makes the tip jump up once (line between 0.20 and 0.25). Not yet verified: the remaining explanations disagree about it and no experiment has decided between them yet.
- Below y = 0.20, using D makes the tip jump down once. Not yet verified: the remaining explanations disagree about it and no experiment has decided between them yet.
- Below y = 0.20, D moves at 2x speed (line between 0.20 and 0.25). Not yet verified: the remaining explanations disagree about it and no experiment has decided between them yet.
- Below y = 0.20, D stops working. Not yet verified: the remaining explanations disagree about it and no experiment has decided between them yet.

## Unsafe to test

- The red zones on the map, plus a 0.03 margin around them: the agent never let the tip go there, so nothing about them was tested.
- Grey areas on the map were never visited with a verified prediction; behaviour there may differ.

## Safe operating area

Green: verified ground. Grey: not visited. Pink: within the margin of a hazard (red). Dashed: lines where a control's behaviour changes. Blue: the path travelled. See manual.html for the map.

## Evidence

- <a id="e38"></a>Experiment 38: D+1 2.00s, tip (0.618, 0.655) → (0.566, 0.647) [frame](frames/0038_before.jpg) [frame](frames/0038_after.jpg)
- <a id="e39"></a>Experiment 39: A+1 B-1 2.00s, tip (0.566, 0.648) → (0.571, 0.582) [frame](frames/0039_before.jpg) [frame](frames/0039_after.jpg)
- <a id="e41"></a>Experiment 41: D+1 1.00s, tip (0.556, 0.581) → (0.507, 0.579) [frame](frames/0041_before.jpg) [frame](frames/0041_after.jpg)
- <a id="e42"></a>Experiment 42: D+1 2.00s, tip (0.507, 0.580) → (0.461, 0.579) [frame](frames/0042_before.jpg) [frame](frames/0042_after.jpg)
