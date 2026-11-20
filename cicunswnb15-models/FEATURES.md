# CIC-UNSW-NB15 features for a benign-vs-anomaly forest

`dataset/Data.csv` joined by row order to `dataset/Label.csv` is the whole
bundle: 76 numeric columns, 447,915 flows. Nothing in this directory is a
cleaned matrix or a split. The measurements below used every row. They are
a prior. The four-column set is a hypothesis to confirm on the training
split.

The task is binary. Code 0 is benign (358,332 flows, 80.0%). Codes 1–9 are
anomaly (89,583 flows, 20.0%). Names are in `dataset/Readme.txt`. Keep the
family code for recall even though the model target is code 0 against the
rest.

| Code | Label | Flows | Share of attacks |
|---|---|---:|---:|
| 0 | Benign | 358,332 | — |
| 4 | Exploits | 30,951 | 34.6% |
| 5 | Fuzzers | 29,613 | 33.1% |
| 7 | Reconnaissance | 16,735 | 18.7% |
| 6 | Generic | 4,632 | 5.2% |
| 3 | DoS | 4,467 | 5.0% |
| 8 | Shellcode | 2,102 | 2.3% |
| 2 | Backdoor | 452 | 0.5% |
| 1 | Analysis | 385 | 0.4% |
| 9 | Worms | 246 | 0.3% |

Exploits and Fuzzers are 67.6% of the attacks. A binary F1 will follow those
two families, so each run also needs recall on Shellcode, Reconnaissance,
Analysis, Backdoor, and Worms.

Tree count, depth, and bit width are open. Plaintext forests try every
depth in 3, 4, 5 and every tree count in 3, 4, 5. Quantization is a second
pass, on the plaintext winner of each feature set, from 8 bits down to 5.
The forest stays small because encrypted inference multiplies the plaintext
cost and the latency budget is under one second.

## Experiment order

1. Join `Data.csv` to `Label.csv` by row order. A full pass found no empty
   cells and no infinities. The label file is the code list in
   `dataset/Readme.txt`. It is not a train/test cut.
2. Drop columns that are a property of the file, before the split. Nine
   columns are identical zeros: `Bwd PSH Flags`, `Fwd URG Flags`,
   `Bwd URG Flags`, `URG Flag Count`, `CWR Flag Count`, `ECE Flag Count`,
   `Fwd Bytes/Bulk Avg`, `Fwd Packet/Bulk Avg`, `Fwd Bulk Rate Avg`. Also
   drop the near-zero columns `RST Flag Count`, `Fwd PSH Flags`, the four
   Active columns, the four Idle columns, and the four Subflow columns.
   Drop the exact duplicates `Fwd Segment Size Avg` (same as
   `Fwd Packet Length Mean`) and `Bwd Segment Size Avg` (same as
   `Bwd Packet Length Mean`).
3. Split rows, stratified on the ten label codes, into train, validation,
   and test. There is no timestamp, address, flow id, or port, so a capture
   or session holdout cannot be built. Flows from the same attack will land
   on both sides. Every score from this split is an optimistic screen.
4. The screen below is the core plus every subset of the three add-ons.
   That is eight matrices: core, each add-on alone, each pair, and all
   three. The add-ons are `Fwd Packet Length Mean`, `Fwd Seg Size Min`, and
   `Bwd Init Win Bytes`. A new column built from some other raw field is a
   later pass. This screen does not reopen the columns dropped in step 2.
5. Transform every matrix the same way. `log1p` on `Bwd Packets/s`,
   `Fwd IAT Mean`, and `Fwd Packet Length Mean`. Clip `Down/Up Ratio` at 6.
   Leave `FIN Flag Count`, `Fwd Seg Size Min`, and `Bwd Init Win Bytes` as
   they are. Fit the min–max on the training rows of that matrix, then clip
   validation and test to `[0, 1]`. The 5-bit F1 numbers later in this note
   used the full-file minimum and maximum, so they stay ceilings for one
   split.
6. Fit a plaintext forest at each depth in 3, 4, 5 and each tree count in
   3, 4, 5. `class_weight="balanced"`, `max_features=1.0`, seed 0. Choose
   the depth and tree count on validation attack F1. Ties go to the lower
   benign false-positive rate, then the smaller forest. Score that winner
   on the test rows after the choice.
7. Refit each winner at 8, 7, 6, and 5 bits and predict in the clear
   (`fhe="disable"`). Choose a bit width from the validation rows. The test
   rows are reported beside that, and they do not pick the bit width. Record
   attack F1, the benign false-positive rate, and recall for Shellcode,
   Reconnaissance, Analysis, Backdoor, and Worms, with the feature set, tree
   count, depth, and bit width. The script is `screen_feature_sets.py`.

## How a column earned a place on the full file

1. It separates benign from anomaly on all 447,915 flows (Mann–Whitney AUC,
   direction folded so 0.5 is chance), including the small attack families.
2. A low bit width can still represent it after a cheap transform. Rates and
   inter-arrival times span many orders of magnitude, and a uniform min–max
   spends its codes on the tail. The tables include a 5-bit stress check:
   the best single threshold on those 32 codes, both directions, whole file.
   Five bits is one point on the bit-width sweep. The always-attack F1 at
   this prevalence is 0.333. These F1 numbers are ceilings for one split,
   and they are not a held-out forest score.
3. It carries information the stronger columns do not repeat. Spearman
   correlations below are from 5,000 benign and 5,000 attack flows.
4. It takes more than one value inside the attack class. Two columns that
   are constant across every attack family have high F1 and are tested on
   their own, because a tree will split on them first.

The transforms in step 5 are the form this prior points to. They are fit
after the split, on training rows only.

## Hypothesis: four columns

These four are the training-split ranking to confirm. The numbers beside
them are from all 447,915 flows.

| Column | Transform | Benign median | Attack median | Attack F1 at 5 bits | Benign false-positive rate |
|---|---|---:|---:|---:|---:|
| `Bwd Packets/s` | `log1p` | 2,281 | 11.0 | 0.758 | 11.8% |
| `Down/Up Ratio` | clip at 6 | 1 | 0 | 0.727 | 12.1% |
| `Fwd IAT Mean` | `log1p` | 11 | 34,260 | 0.666 | 13.9% |
| `FIN Flag Count` | none | 1 | 1 | 0.451 | 3.9% |

The attack F1 is not a forest score. For each column the transformed values
are min–max scaled on all 447,915 flows and rounded to 32 codes
(`round((x - min) / (max - min) × 31)`). Every cut is tried in both
directions, “attack if the code is above the cut” and “attack if the code
is at or below the cut.” Precision and recall are for the attack class, and
the table keeps the cut with the highest F1. The same rows choose the cut
and receive the score, so the number is an in-sample ceiling for a single
split. The false-positive rate is benign flows on the attack side of that
same cut, divided by 358,332.

| Column | Winning cut | Attack recall |
|---|---|---:|
| `Bwd Packets/s` | `log1p` code ≤ 11 | 0.898 |
| `Down/Up Ratio` | clipped value is 0 | 0.848 |
| `Fwd IAT Mean` | `log1p` code > 16 | 0.776 |
| `FIN Flag Count` | count > 1 | 0.337 |

### `Bwd Packets/s`

Backward packet rate is the only column at AUC 0.94 or better against every
attack family:

| Family | AUC |
|---|---:|
| Analysis | 0.984 |
| Fuzzers | 0.960 |
| Reconnaissance | 0.959 |
| Shellcode | 0.959 |
| Generic | 0.948 |
| Backdoor | 0.947 |
| DoS | 0.943 |
| Exploits | 0.939 |
| Worms | 0.939 |

Binary AUC is 0.951. Attacks send about two orders of magnitude less
backward traffic than benign flows. At 5 bits the raw column still
separates (attack F1 0.692) and piles attacks into the bottom code, at a
benign false-positive rate of 22.2%. `log1p` spreads those 32 codes across
the real range: F1 0.758, recall 0.898, benign false-positive rate 11.8%.
The prior stores this column as `log1p`. Bit width is part of the later
sweep, and the scaler is fit on training rows only.

### `Down/Up Ratio`

The ratio of backward to forward packets says the same thing in a form a
tree can split exactly. Benign flows are 46.4% ratio 1 and 41.2%
ratio 2. Every attack family’s median is 0. Fuzzers are 97.8% zero,
Reconnaissance 99.9% zero, Shellcode 100% zero. Binary AUC is 0.868.
Family AUC is 0.93–0.94 on Fuzzers, Reconnaissance, and Shellcode, and
0.76–0.78 on Generic and Exploits, which is where forward inter-arrival
time takes over.

The column has a handful of observed integers, then seven rows at 398 or
400 (one at 398, six at 400). A raw min–max lets those seven rows set the
scale. At 5 bits that collapses codes 0–6 into one code and the attack F1
falls to 0.333. Clipping at 6 before the scaler keeps one code per real
value and restores attack F1 0.727, recall 0.848, benign false-positive
rate 12.1%. `log1p` matches that F1 on this column. The clip is the form
to store, because the values are already small integers. At a higher bit
width the seven tail rows matter less; they still should not set the scale.

Spearman correlation with `Bwd Packets/s` is 0.75. The two columns agree
that anomaly flows are one-directional, and they disagree often enough that
both belong in a four-input tree.

### `Fwd IAT Mean`

Forward inter-arrival time catches the families the ratio is softer on.
Median 11 on benign flows and 34,260 on attacks. Binary AUC is 0.860.
Analysis is 0.974, Exploits 0.922, Backdoor 0.908. Shellcode is the weak
family at 0.743, which is still well above the flow-level packet rate on
that family.

The maximum is 4.2×10^7, so a raw min–max spends its codes on the tail. At
5 bits the raw attack F1 is 0.333. `log1p` brings that stress check to
0.666, recall 0.776, benign false-positive rate 13.9%.

`Flow IAT Mean` scores 0.690 on the same 5-bit check after `log1p`. It
loses Shellcode: AUC 0.595, against 0.743 for forward inter-arrival. One
timing column is enough. On the 10,000-flow sample, Spearman |r| with
`Fwd IAT Mean` is 0.93 for `Flow Packets/s` and `Fwd Packets/s`, and 0.91
for `Flow Duration`. The other IAT means, standard deviations, maxima, and
totals sit in the same cluster. `Flow Packets/s` is also weak on Shellcode
(AUC 0.59) and Reconnaissance (AUC 0.67).

Spearman correlation of `Fwd IAT Mean` with `Bwd Packets/s` is −0.72, so
the timing column is related to the rate column and still moves when the rate does not.

### `FIN Flag Count`

FIN is the fourth input because it is nearly independent of the rate
(Spearman −0.09 with `Bwd Packets/s`) and the false-positive rate stays
low. The column takes values 0, 1, 2, and 3, so any bit width that can
hold four codes keeps them apart. The split is “attack if the
count is above 1”: attack F1 0.451, recall 33.7%, benign false-positive
rate 3.9%. The medians are both 1 because the signal is in that tail.
Benign flows are 79.7% count 1. Analysis is 98.7% count 2 (family AUC
0.974). Exploits are 0.805. Fuzzers and Reconnaissance are about 0.53, so
FIN belongs as a companion split beside the rate columns.

## Hypothesis: the three add-ons, in every subset

The screen fits the core with every subset of these three columns, pairs
and the triple included. A single add-on shows what that column does beside
the core. A pair or the triple shows what still moves once another add-on
is already in the matrix. Segment size and the backward window are the
columns a tree will split on first, so a chain that always adds them in one
order would hide the others.

### `Fwd Packet Length Mean`, with `log1p`

Binary AUC is 0.648. After `log1p`, a single 5-bit split reaches attack F1
0.427 at a benign false-positive rate of 35.8%. That is a weak split on
its own, at the low end of the bit-width sweep.
It is the payload-shape column least tied to the rate (Spearman −0.56 with
`Bwd Packets/s`). Family AUC is 0.813 on Analysis and 0.770 on Generic, and
0.517 on Backdoor. The extra run asks whether size still helps those two
families once the rate features are present.

`Fwd Segment Size Avg` is the same column. `Bwd Segment Size Avg` is the
same column as `Bwd Packet Length Mean`. Keep one of each pair.

### `Fwd Seg Size Min`, unchanged

Seven values, from 8 to 40. Any bit width that can hold seven codes keeps
them apart; at 5 bits each value already has its own code. Attack F1 of
one split is 0.729, recall 99.96%, benign
false-positive rate 18.6%. The rule is “attack if the value is below 24”.

| Value | Flows | Benign | Attack |
|---:|---:|---:|---:|
| 32 | 291,607 | 291,607 | 0 |
| 20 | 85,505 | 9,136 | 76,369 |
| 8 | 70,515 | 57,341 | 13,174 |
| 24 | 228 | 228 | 0 |
| 28 | 43 | 3 | 40 |
| 36 | 16 | 16 | 0 |
| 40 | 1 | 1 | 0 |

Value 32 is 81.4% of benign flows and never occurs on an attack. Every
attack family’s median is 20. A tree that can see this column will split
on it first, at any depth that has a spare cut. The add-on run measures how much of the score is
that code. If the score jumps here and falls back on the four-column run,
the forest is reading a capture constant.

### `Bwd Init Win Bytes`, unchanged

Binary AUC is 0.718. At 5 bits the raw single-split attack F1 is 0.657,
recall 0.853, benign false-positive rate 18.6%. `log1p` drops that F1 to
0.424, so the add-on uses the raw values. Every attack family’s median is
16,384. The benign median is 14,480. Same kind of check as segment size:
the add-on run measures how much of the score is the attack window.
`FWD Init Win Bytes` repeats the same idea more weakly (AUC 0.673; 5-bit
single-split F1 0.589) and stays out.

## Columns left out

| Columns | Reason |
|---|---|
| `Bwd PSH Flags`, `Fwd URG Flags`, `Bwd URG Flags`, `URG Flag Count`, `CWR Flag Count`, `ECE Flag Count`, `Fwd Bytes/Bulk Avg`, `Fwd Packet/Bulk Avg`, `Fwd Bulk Rate Avg` | Nine columns, all identical, all zero. |
| `RST Flag Count`, `Fwd PSH Flags` | 99.99% and 99.93% zero. RST attack F1 is 0. |
| `Active Mean`, `Active Std`, `Active Max`, `Active Min`, `Idle Mean`, `Idle Std`, `Idle Max`, `Idle Min` | At least 99.5% zero. Both class medians are 0. |
| `Subflow Fwd Packets`, `Subflow Fwd Bytes`, `Subflow Bwd Packets`, `Subflow Bwd Bytes` | At least 96.9% zero. In this export they are not copies of the total packet and byte columns. |
| `Flow Packets/s`, `Fwd Packets/s`, `Flow Duration`, and the IAT columns other than `Fwd IAT Mean` | Spearman \|r\| of 0.90 or more with `Fwd IAT Mean`, or with each other inside that cluster. |
| `Total Length of Fwd Packet` | A single split stays weak. At 5 bits the raw attack F1 is 0.333. After `log1p` it is 0.386, at a benign false-positive rate of 43.8%. |
| `ACK Flag Count`, `SYN Flag Count`, `PSH Flag Count` | A log or a clip that keeps the small counts still leaves a single split weak. At 5 bits the attack F1 is 0.476, 0.366, and 0.414, with benign false-positive rates of 43.2%, 41.4%, and 60.7%. |
| `Flow Bytes/s`, `Packet Length Min` | Binary AUC 0.52 and 0.50. `Flow Bytes/s` reaches 1.47×10^9. |

## What the screen writes

`screen_feature_sets.py` writes `screen_runs/<local timestamp>/`.
`run.log` records every fit as it starts and as it finishes, and the csv
for that phase grows by one row at the same time. `plaintext.csv` is the
validation grid. `winners.csv` is one shape per feature set.
`plaintext_test.csv` scores those shapes after the choice. `quantized.csv`
is 8, 7, 6, and 5 bits on validation and test. `report.txt` prints the
same rows. `arguments.json` and `status.json` mark the run. The split
remains an optimistic screen, because nothing in this file identifies a
capture.
