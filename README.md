# cicevse-session-audit

Reproducibility repository for the protocol audit of session confounding in
host-based intrusion detection on CICEVSE2024 electric-vehicle charging
infrastructure.

The central result is that random-row evaluation substantially overstates
detection performance relative to capture-session-disjoint evaluation. Under
the primary XGBoost protocol, matched 15-class macro-F1 is 0.787 for random
splits and 0.199 for session-disjoint splits, a gap of 0.589. LightGBM gives
the same conclusion (0.787 versus 0.218; gap 0.569).

## What the audit does

CICEVSE2024 host events were captured in 36 sessions. Many attack classes occur
in only one or two sessions, and most sessions contain a single class. A
random-row split therefore places events from the same capture session in both
training and test data. The audit measures the difference between random-row
and capture-session-disjoint evaluation, with and without inverse-frequency
class weighting.

All four conditions use the same prepared features, model settings, and five
seeds (42, 0, 1, 2, 3). Macro-F1 is computed over the 15 classes present in the
session-disjoint test folds so that random and session results use an
identical class set. Classes that occur in only one session (`os-scan` and
`icmp-fragmentation_old`) are evaluated separately as a one-session probe.

## Results

Matched macro-F1, mean ± standard deviation over five seeds:

| Condition | Split | Weighting | XGBoost | LightGBM |
| --- | --- | --- | --- | --- |
| E1 | random | standard | 0.787 ± 0.016 | 0.787 ± 0.007 |
| E2 | random | inverse frequency | 0.787 ± 0.014 | 0.788 ± 0.015 |
| E3 | session-disjoint | standard | 0.199 ± 0.079 | 0.218 ± 0.053 |
| E4 | session-disjoint | inverse frequency | 0.193 ± 0.051 | 0.217 ± 0.048 |

| Contrast | XGBoost gap | LightGBM gap |
| --- | --- | --- |
| E1 − E3 | 0.589 ± 0.093 | 0.569 ± 0.055 |
| E2 − E4 | 0.594 ± 0.059 | 0.571 ± 0.054 |

The gap is positive for all five seeds in both model families. Inverse-frequency
weighting does not close it.

## Repository layout

```text
cicevse-session-audit/
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
│   ├── experiment_config.yaml
│   └── run_matrix.csv
├── data/
│   ├── README.md
│   ├── session_assignments.csv
│   ├── label_repair_evidence.csv
│   └── raw/                  # place the downloaded dataset here
├── src/
│   ├── data_prep/
│   ├── splits/
│   ├── models/
│   └── evaluation/
├── results/
│   ├── xgboost/
│   │   ├── predictions/
│   │   └── metrics/
│   ├── lightgbm/
│   │   ├── predictions/
│   │   └── metrics/
│   └── figures/
├── tests/
└── docs/
    └── DATA_AVAILABILITY.md
```

## Data

The raw CICEVSE2024 host-event export is not included. Download it from the
Canadian Institute for Cybersecurity CICEVSE2024 distribution and place the
combined host-event file at:

```text
data/raw/EVSE-B-HPC-Kernel-Events-Combined.csv
```

The expected SHA-256 is:

```text
0025e946bb0c7959752df00df3b516012faabddbde0644577d5fb732963de553
```

The raw table has 8,474 rows and 915 columns. The preparation pipeline produces
8,468 rows, 308 features, 36 sessions, and 18 classes. The committed
`data/session_assignments.csv` documents every session; the committed
`data/label_repair_evidence.csv` records the two affected backdoor sessions.
See [`data/README.md`](data/README.md) for details.

## Installation

Python 3.12.6 is the reference environment.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Verify the committed results without retraining

The saved prediction files are sufficient to re-score every run and check the
headline gaps.

```bash
python -m src.evaluation.recompute_metrics
python -m pytest -q
```

Expected output includes:

```text
xgboost: standard gap=0.589 +/- 0.093; weighted gap=0.594 +/- 0.059
lightgbm: standard gap=0.569 +/- 0.055; weighted gap=0.571 +/- 0.054

16 passed
```

The tests cover the label repair, session reconstruction, split construction,
class weighting, metric handling, the 40-run matrix, and the headline gaps.

## Full reproduction

Run from the repository root after placing the raw file in `data/raw/`.

```bash
python -m src.data_prep.data_prep

python -m src.models.run_xgboost
python -m src.evaluation.analyze_xgboost

python -m src.models.run_lightgbm
python -m src.evaluation.analyze_lightgbm

python -m src.evaluation.recompute_metrics
python -m src.evaluation.make_figures
python -m pytest -q
```

The commands regenerate the derived data tables, all 20 XGBoost runs, all 20
LightGBM runs, per-run predictions, per-class metrics, confusion matrices,
coverage tables, matched-class comparisons, session diagnostics, and the five
figures in `results/figures/`. Full retraining takes substantially longer than
the verification path and overwrites the committed result files.

## Method summary

- **Session reconstruction.** A new session identifier is assigned whenever the
  timestamp sequence moves backwards. The resulting 36 sessions are recorded in
  `data/session_assignments.csv`.
- **Label repair.** 2,302 backdoor rows have attack fields displaced one column
  to the right. The repair maps the fields back to the backdoor class before
  modelling and preserves a row-level evidence table.
- **Random split.** Stratified 60/20/20 row split.
- **Session-disjoint split.** Whole sessions are held out for classes with two
  or three sessions. Classes with two sessions contribute the last 20% of the
  training session to validation. Session 0, which contains benign and
  cryptojacking traffic, is divided by time into 60/20/20 partitions.
- **Imputation.** Median imputation is fitted on the training partition only.
- **Class weighting.** Inverse-frequency weights computed from the training
  partition only: `w_c = n_train / (n_classes * count_c_train)`.
- **Models.** XGBoost and LightGBM with 400-tree budgets, depth 6, learning rate
  0.10, and early stopping on the validation partition.
- **Metrics.** Macro-F1 over the 15 matched classes; balanced accuracy,
  accuracy, weighted F1, per-class precision/recall/F1, and confusion matrices
  are also saved.
- **Session diagnostic.** A model predicts session identity from host features
  on the random-split training partition (seed 42). The result is reported
  against a majority-session baseline and a session-by-class contingency table
  because session identity and attack class are strongly confounded.



## License

Released under the MIT License. See [`LICENSE`](LICENSE). The raw CICEVSE2024
dataset remains subject to the terms of its original distribution.

## Citation

```bibtex

```
