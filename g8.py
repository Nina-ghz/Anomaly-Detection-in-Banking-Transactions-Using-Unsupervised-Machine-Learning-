import pandas as pd
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

PREDICTIONS_FILE = "injected_predictions_with_ensemble.csv"
TRANSACTIONS_FILE = "../injections/transactions_client_final_injected325.csv"

OUTPUT_EVALUATED = "injected_predictions_evaluated11.csv"
OUTPUT_RESULTS = "model_evaluation_results11.csv"


# These are the ONLY rows considered injected anomalies.
# SMURFING_RAPID_BURST is intentionally excluded.
ANOMALY_TYPES = [
    "NIGHT_RAPID_BURST",
    "NIGHT_LARGE_AMOUNT",
    "UNUSUAL_SPENDING",
    "FOREIGN_SPIKE",
    "FOREIGN_NIGHT",
]


# ============================================================
# 1. LOAD FILES
# ============================================================

pred = pd.read_csv(PREDICTIONS_FILE)

transactions = pd.read_csv(
    TRANSACTIONS_FILE,
    encoding="utf-8-sig",
    low_memory=False
)

print("=" * 80)
print("FILES")
print("=" * 80)

print(f"Prediction rows   : {len(pred):,}")
print(f"Transaction rows  : {len(transactions):,}")


# ============================================================
# 2. CHECK PREDICTION FILE
# ============================================================

EXPECTED_PREDICTION_ROWS = 652526

if len(pred) != EXPECTED_PREDICTION_ROWS:
    raise ValueError(
        f"Unexpected prediction row count: {len(pred):,}. "
        f"Expected {EXPECTED_PREDICTION_ROWS:,}."
    )


# ============================================================
# 3. CHECK INJECTION LABELS BEFORE REMOVAL
# ============================================================

print("\n" + "=" * 80)
print("INJECTION LABELS BEFORE SMURFING REMOVAL")
print("=" * 80)

print(transactions["INJECTED"].value_counts(dropna=False))


print("\nANOMALY TYPE DISTRIBUTION:")
print(
    transactions["ANOMALY_TYPE"]
    .value_counts(dropna=False)
    .sort_index()
)


# ============================================================
# 4. REMOVE SMURFING FROM EVALUATION
#
# We completely remove SMURFING_RAPID_BURST rows.
#
# The predictions file still contains predictions for them,
# but they will not be included in the evaluation after merge.
# ============================================================

SMURFING_TYPE = "SMURFING_RAPID_BURST"

before_count = len(transactions)

transactions = transactions[
    transactions["ANOMALY_TYPE"] != SMURFING_TYPE
].copy()

after_count = len(transactions)

smurfing_removed = before_count - after_count

print("\n" + "=" * 80)
print("SMURFING REMOVED FROM EVALUATION")
print("=" * 80)

print(f"Rows before removal      : {before_count:,}")
print(f"Smurfing rows removed    : {smurfing_removed:,}")
print(f"Rows after removal       : {after_count:,}")

if smurfing_removed != 60:
    raise ValueError(
        f"Expected 60 smurfing rows to be removed, "
        f"but removed {smurfing_removed:,}."
    )


# ============================================================
# 5. BUILD CORRECT GROUND TRUTH
#
# INJECTED = 1 includes:
#   - actual anomalies
#   - BASELINE_NORMAL support transactions
#
# Therefore we DO NOT use INJECTED as ground truth.
#
# Ground truth = 1 only for the five evaluated anomaly types.
# ============================================================

transactions["GROUND_TRUTH"] = (
    transactions["ANOMALY_TYPE"].isin(ANOMALY_TYPES)
).astype(int)


# ============================================================
# 6. VERIFY GROUND TRUTH
# ============================================================

n_anomalies = int(
    transactions["GROUND_TRUTH"].sum()
)

n_normal = int(
    (transactions["GROUND_TRUTH"] == 0).sum()
)

print("\n" + "=" * 80)
print("GROUND TRUTH VERIFIED")
print("=" * 80)

print(f"Total transactions : {len(transactions):,}")
print(f"Evaluated anomalies: {n_anomalies:,}")
print(f"Normal transactions: {n_normal:,}")

if n_anomalies != 265:
    raise ValueError(
        f"Expected exactly 265 evaluated anomalies, "
        f"got {n_anomalies:,}."
    )

expected_evaluated_rows = 652466

if len(transactions) != expected_evaluated_rows:
    raise ValueError(
        f"Expected {expected_evaluated_rows:,} transactions "
        f"after removing smurfing, got {len(transactions):,}."
    )


# ============================================================
# 7. CREATE MATCHING KEY
#
# We use:
#   TRS_NO + TRS_AC_CIF + DATETIME
#
# to safely associate predictions with transactions.
# ============================================================

for df in [pred, transactions]:

    df["TRS_NO"] = df["TRS_NO"].astype(str)
    df["TRS_AC_CIF"] = df["TRS_AC_CIF"].astype(str)
    df["DATETIME"] = df["DATETIME"].astype(str)

    df["_KEY"] = (
        df["TRS_NO"] + "_"
        + df["TRS_AC_CIF"] + "_"
        + df["DATETIME"]
    )


# ============================================================
# 8. CHECK KEY UNIQUENESS
# ============================================================

if transactions["_KEY"].duplicated().any():

    duplicates = transactions["_KEY"].duplicated().sum()

    raise ValueError(
        f"Transaction key is not unique. "
        f"Found {duplicates:,} duplicate keys."
    )


if pred["_KEY"].duplicated().any():

    duplicates = pred["_KEY"].duplicated().sum()

    raise ValueError(
        f"Prediction key is not unique. "
        f"Found {duplicates:,} duplicate keys."
    )


# ============================================================
# 9. MERGE GROUND TRUTH WITH PREDICTIONS
#
# Because smurfing was removed from transactions,
# smurfing predictions will NOT appear in evaluated.
# ============================================================

ground_truth = transactions[
    [
        "_KEY",
        "TRS_NO",
        "TRS_AC_CIF",
        "DATETIME",
        "ANOMALY_TYPE",
        "GROUND_TRUTH",
    ]
].copy()


evaluated = pred.merge(
    ground_truth,
    on="_KEY",
    how="inner",
    suffixes=("", "_GT"),
    validate="one_to_one"
)


# ============================================================
# 10. VERIFY MERGE
# ============================================================

if len(evaluated) != len(transactions):
    raise ValueError(
        f"Merge mismatch: expected {len(transactions):,} "
        f"evaluated rows, got {len(evaluated):,}."
    )


if evaluated["GROUND_TRUTH"].isna().any():

    missing = evaluated["GROUND_TRUTH"].isna().sum()

    raise ValueError(
        f"{missing:,} prediction rows could not be matched "
        f"with transaction ground truth."
    )


evaluated["GROUND_TRUTH"] = (
    evaluated["GROUND_TRUTH"].astype(int)
)


# Make sure smurfing is completely absent
if (
    evaluated["ANOMALY_TYPE"] == SMURFING_TYPE
).any():

    raise ValueError(
        "SMURFING_RAPID_BURST is still present in evaluated data."
    )


print("\nMerge successful.")
print(f"Evaluated rows: {len(evaluated):,}")
print(f"Evaluated anomalies: {evaluated['GROUND_TRUTH'].sum():,}")
print(
    f"Evaluated normal: "
    f"{(evaluated['GROUND_TRUTH'] == 0).sum():,}"
)


# ============================================================
# 11. MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    prediction_column,
    score_column
):

    y_true = (
        evaluated["GROUND_TRUTH"]
        .values
    )

    y_pred = (
        evaluated[prediction_column]
        .astype(int)
        .values
    )

    scores = (
        evaluated[score_column]
        .astype(float)
        .values
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    # --------------------------------------------------------
    # AUC
    #
    # Isolation Forest:
    # lower decision_function = more anomalous
    #
    # Therefore invert the score.
    # --------------------------------------------------------

    if model_name == "Isolation Forest":
        auc_scores = -scores
    else:
        auc_scores = scores

    roc_auc = roc_auc_score(
        y_true,
        auc_scores
    )

    pr_auc = average_precision_score(
        y_true,
        auc_scores
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return {
        "MODEL": model_name,
        "PRECISION": precision,
        "RECALL": recall,
        "F1_SCORE": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }


# ============================================================
# 12. EVALUATE ALL MODELS
# ============================================================

results = []


# ------------------------------------------------------------
# Isolation Forest
# ------------------------------------------------------------

results.append(
    evaluate_model(
        "Isolation Forest",
        "IF_ANOMALY",
        "IF_SCORE"
    )
)


# ------------------------------------------------------------
# LOF
# ------------------------------------------------------------

results.append(
    evaluate_model(
        "LOF",
        "LOF_ANOMALY",
        "LOF_SCORE"
    )
)


# ------------------------------------------------------------
# Autoencoder
# ------------------------------------------------------------

results.append(
    evaluate_model(
        "Autoencoder",
        "AE_ANOMALY",
        "AE_SCORE"
    )
)


# ------------------------------------------------------------
# Ensemble
# ------------------------------------------------------------

results.append(
    evaluate_model(
        "Ensemble (>=2)",
        "ENSEMBLE_ANOMALY",
        "N_MODELS_FLAGGED"
    )
)


# ============================================================
# 13. RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("MODEL PERFORMANCE")
print("=" * 80)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# 14. DETECTION RATE
# ============================================================

print("\n" + "=" * 80)
print("DETECTION RATE ON 265 EVALUATED ANOMALIES")
print("=" * 80)

total_anomalies = int(
    evaluated["GROUND_TRUTH"].sum()
)

for model_name, prediction_column in [

    ("Isolation Forest", "IF_ANOMALY"),
    ("LOF", "LOF_ANOMALY"),
    ("Autoencoder", "AE_ANOMALY"),
    ("Ensemble (>=2)", "ENSEMBLE_ANOMALY"),

]:

    detected = int(
        evaluated.loc[
            evaluated["GROUND_TRUTH"] == 1,
            prediction_column
        ].sum()
    )

    rate = (
        detected / total_anomalies * 100
    )

    print(
        f"{model_name:<20}: "
        f"{detected:>3}/{total_anomalies} "
        f"({rate:.2f}%)"
    )


# ============================================================
# 15. DETECTION BY ANOMALY TYPE
# ============================================================

print("\n" + "=" * 80)
print("DETECTION BY ANOMALY TYPE")
print("=" * 80)

anomaly_rows = evaluated[
    evaluated["ANOMALY_TYPE"].isin(ANOMALY_TYPES)
].copy()


type_results = []


for anomaly_type in ANOMALY_TYPES:

    subset = anomaly_rows[
        anomaly_rows["ANOMALY_TYPE"] == anomaly_type
    ]

    total = len(subset)

    if total == 0:
        continue

    row = {
        "ANOMALY_TYPE": anomaly_type,
        "TOTAL": total,
    }

    for model_name, prediction_column in [

        ("IF", "IF_ANOMALY"),
        ("LOF", "LOF_ANOMALY"),
        ("AE", "AE_ANOMALY"),
        ("ENSEMBLE", "ENSEMBLE_ANOMALY"),

    ]:

        detected = int(
            subset[prediction_column].sum()
        )

        row[
            f"{model_name}_DETECTED"
        ] = detected

        row[
            f"{model_name}_RATE"
        ] = detected / total * 100

    type_results.append(row)


type_results_df = pd.DataFrame(
    type_results
)


print(
    type_results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ============================================================
# 16. SHOW BASELINE SUPPORT ROWS
#
# These rows are NOT anomalies.
# We show them separately.
# ============================================================

baseline_rows = evaluated[
    evaluated["ANOMALY_TYPE"] == "BASELINE_NORMAL"
]


print("\n" + "=" * 80)
print("BASELINE SUPPORT TRANSACTIONS")
print("=" * 80)

print(
    f"Baseline support rows: "
    f"{len(baseline_rows):,}"
)


for model_name, prediction_column in [

    ("Isolation Forest", "IF_ANOMALY"),
    ("LOF", "LOF_ANOMALY"),
    ("Autoencoder", "AE_ANOMALY"),
    ("Ensemble", "ENSEMBLE_ANOMALY"),

]:

    flagged = int(
        baseline_rows[prediction_column].sum()
    )

    print(
        f"{model_name:<20}: "
        f"{flagged:,} flagged"
    )


# ============================================================
# 17. VERIFY SMURFING IS ABSENT
# ============================================================

smurfing_in_evaluation = (
    evaluated["ANOMALY_TYPE"]
    == SMURFING_TYPE
).sum()


print("\n" + "=" * 80)
print("FINAL SMURFING CHECK")
print("=" * 80)

print(
    f"SMURFING_RAPID_BURST rows in evaluation: "
    f"{smurfing_in_evaluation}"
)

if smurfing_in_evaluation != 0:
    raise ValueError(
        "Smurfing rows are still present in evaluation."
    )


# ============================================================
# 18. SAVE EVALUATED DATA
# ============================================================

evaluated.to_csv(
    OUTPUT_EVALUATED,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 19. SAVE MODEL RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_RESULTS,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    "Evaluated predictions : "
    f"{OUTPUT_EVALUATED}"
)

print(
    "Model results         : "
    f"{OUTPUT_RESULTS}"
)

print("\n" + "=" * 80)
print("FINAL EVALUATION SUMMARY")
print("=" * 80)

print(f"Original transactions       : {before_count:,}")
print(f"Smurfing removed            : {smurfing_removed:,}")
print(f"Final evaluated transactions: {len(evaluated):,}")
print(f"Actual anomalies evaluated  : {n_anomalies:,}")
print(f"Normal transactions         : {n_normal:,}")

print("\nEvaluation complete.")
print("=" * 80)