# Transaction Anomaly Detection

A Python pipeline for detecting anomalous banking transactions from CSV data. It combines three unsupervised models (Isolation Forest, Local Outlier Factor, PyTorch Autoencoder) with an ensemble vote and explicit business rules, and is evaluated on synthetic anomalies injected into the data.

## Overview

The project works on the **transactions table only**. An early design linked transactions, accounts and clients into a single merged file, but that approach was dropped in favour of training models directly on engineered transaction-level features.

Target anomaly patterns:

| Category | Signal |
|---|---|
| `UNUSUAL_SPENDING` | Amount far above the client's usual baseline |
| `NIGHT_LARGE_AMOUNT` | Night-time transaction + amount spike |
| `NIGHT_RAPID_BURST` | Night-time + amount spike + many transactions in 24h |
| `SMURFING_RAPID_BURST` | High transaction counts over 24h and 7 days |
| `FOREIGN_SPIKE` | Amount spike on a foreign-currency / exchange-rate transaction |
| `FOREIGN_NIGHT` | Foreign-currency transaction at night |

## Pipeline

```
Cleaned CSVs ──► Feature engineering ──► Model training ──► Ensemble (>= 2 models) ──► Review priority
                        │                                          ▲
                        └──► Synthetic anomaly injection ──► Evaluation (t7.py)
```

1. **Data cleaning** – the transactions, accounts and clients files were cleaned beforehand (`history_cleaned`, `accounts_cleaned_Final`, `clients_cleaned_Final`, plus lookup tables `cif_type_cleaned` and `ledger_cleaned`). The transactions table is split into several CSV parts.
2. **Feature engineering** – builds per-client behavioural features from the transactions, keeping all original columns.
3. **Model training** – trains Isolation Forest, LOF and an Autoencoder with fixed thresholds.
4. **Ensemble** – a transaction is flagged when at least 2 of the 3 models flag it.
5. **Evaluation** – 325 synthetic anomalies are injected and scored with the already-trained models and scalers.

## Features (11)

| Feature | Description |
|---|---|
| `AMOUNT` | Transaction amount |
| `CLIENT_AVG_AMOUNT` | Client's historical average amount (based on `CV_AMOUNT`) |
| `IS_NIGHT` | Night-time flag |
| `TRS_COUNT_24H` | Transactions in the last 24h |
| `TRS_COUNT_7D` | Transactions in the last 7 days |
| `IS_FOREIGN_CURRENCY` | Foreign-currency flag |
| `RATIO` | Amount relative to client average (baseline floor 1.0, capped at 100) |
| `AMOUNT_ZSCORE` | Per-client z-score of the amount |
| `COUNT_RATIO` | Transaction frequency relative to the client's norm |
| `AMOUNT_SUM_24H` | Rolling 24h sum of the converted amount |
| `MEDIAN_RATIO` | Amount relative to the client's historical median (same floor/cap logic) |

> **Note:** `CLIENT_AVG_AMOUNT` and `RATIO` are computed from `CV_AMOUNT` (the amount converted to local currency), not the raw `AMOUNT`. Using the raw amount distorted foreign-currency transactions.

## Models and thresholds

| Model | Script | Flagging rule |
|---|---|---|
| Isolation Forest | `iso7.py` | Bottom 5% of `decision_function` |
| Local Outlier Factor | `lof7.py` | Top 3% (p97) |
| Autoencoder (PyTorch) | `auuto7.py` | Top 5% reconstruction error (p95) |

Thresholds used when applying the trained models to injected data (`t7.py`): IF = `0.100224`, LOF = `-1.429081`, AE = `0.9505`.

## Rule-based flags

Because the spike and periodic patterns make up a large share of the data (~3.2% and ~1.9%), pure ML with a 1% contamination level missed most of them. A hybrid approach is used:

- **Amount spike:** `RATIO_TO_AVG >= 10x` (baseline min 1.0, ratio capped at 100)
- **Periodic pattern:** `REPEATING_STREAK >= 5` consecutive transactions

A manual review of 86 sampled transactions showed the periodic-pattern rule mostly catches legitimate recurring payments (0% genuine, 80% false positive), so it was dropped from the final ML feature set.

## Review prioritisation

`ensemble9.py` merges model outputs and rule flags (inner join on `BRANCH_CODE`, `TRS_NO`, `DATETIME`, `TRS_AC_CIF`), checks for duplicates, computes `N_MODELS_FLAGGING`, and assigns `REVIEW_PRIORITY` (urgent / high / medium / low / none).

On 650,432 transactions:

| Models agreeing | Transactions |
|---|---|
| 3 | 38 |
| 2 | 1,600 |
| 1 | 16,201 |
| 0 | 632,593 |

Priority breakdown: urgent = 9, high = 745, medium = 4,157, low = 12,928, none = 632,593.

## Synthetic anomaly injection

`id.py` and `trans.py` (later versions generate 325 anomalies) inject labelled synthetic transactions (`IS_INJECTED`, `ANOMALY_TYPE`) so detection can be measured.

| Category | Count |
|---|---|
| `NIGHT_RAPID_BURST` | 60 |
| `NIGHT_LARGE_AMOUNT` | 60 |
| `SMURFING_RAPID_BURST` | 60 |
| `UNUSUAL_SPENDING` | 60 |
| `FOREIGN_SPIKE` | 45 |
| `FOREIGN_NIGHT` | 40 |
| **Total** | **325** |

Injected rows are easy to isolate: `BRANCH_CODE = 10000`, `DATETIME` in 2027 or later, `TRS_AC_CIF` starting with `777`, `TRS_NO` starting with `9999`.

## Results (325 injected anomalies)

| Model | Recall | Comment |
|---|---|---|
| Isolation Forest | 100% | Near-zero precision (645,696 false positives) |
| LOF | 25.54% | |
| Autoencoder | 59.08% | |
| **Ensemble (>= 2 models)** | **71.08%** | Best balance |

Earlier autoencoder experiments (10 features, P98 threshold) showed `FOREIGN_SPIKE` detected at 100%, `FOREIGN_NIGHT` ~72.5% and `NIGHT_LARGE_AMOUNT` ~46.7%, while `NIGHT_RAPID_BURST`, `SMURFING_RAPID_BURST` and `UNUSUAL_SPENDING` stayed at 0% regardless of threshold. The cause was the reconstruction error being averaged (diluted) across features rather than a threshold issue; using the per-feature **max** error instead of the mean was identified as the next step.

## Repository layout

| File | Purpose |
|---|---|
| `feature_engineering_v2.py` / `feature5.py` | Feature engineering on real transactions |
| `id.py`, `trans.py` | Synthetic anomaly injection and its features |
| `iso7.py`, `lof7.py`, `auuto7.py` | Model training (final versions) |
| `c7.py` | Ensemble fusion (>= 2 models) |
| `t7.py` | Apply trained models and scalers to injected data |
| `ensemble9.py` | Merge models + rules, compute review priority |

Generated artefacts include `transactions_ids*.csv`, `transactions_features_for_scaling*.csv`, the trained models and scalers (including `autoencoder_model.pt`).

## Usage

```bash
# 1. Generate features for real transactions
python feature5.py

# 2. Train the models
python iso7.py
python lof7.py
python auuto7.py

# 3. Fuse results (>= 2 models)
python c7.py

# 4. Evaluate on injected anomalies (uses trained models + scalers)
python t7.py
```

Adjust file names and paths to match your local setup.

## Requirements

- Python 3.x
- pandas, numpy, scikit-learn
- PyTorch (autoencoder)

## Next steps

- Replace the mean reconstruction error with the per-feature max in the autoencoder.
- Improve Isolation Forest precision (currently flags nearly everything in evaluation).
- Review `unsure` cases from the manual audit to refine rules.
