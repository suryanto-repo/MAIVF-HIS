import argparse
import os
import random
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import Ridge

from gates.data_quality import evaluate_data_quality
from gates.performance import regression_metrics, normalized_regression_score
from gates.temporal_robustness import evaluate_temporal_robustness
from gates.calibration import evaluate_calibration
from gates.uncertainty_explainability import (
    evaluate_uncertainty_explainability
)
from gates.his_integration import evaluate_his_integration
from degradation.controlled_degradation import (
    generate_degradation_conditions
)


# ================================================================
# ARGUMENTS
# ================================================================

parser = argparse.ArgumentParser(
    description="MAIVF-HIS reference validation workflow"
)

parser.add_argument(
    "--data",
    required=True,
    help="Local .xlsx validation dataset (not committed to repository)"
)

parser.add_argument(
    "--config",
    default="config/validation_config.yaml"
)

args = parser.parse_args()


# ================================================================
# CONFIGURATION
# ================================================================

with open(
    args.config,
    "r",
    encoding="utf-8"
) as f:
    cfg = yaml.safe_load(f)


seed = cfg["reproducibility"]["random_seed"]
target = cfg["task"]["target"]

random.seed(seed)
np.random.seed(seed)


outdir = "outputs/validation_report"

os.makedirs(
    outdir,
    exist_ok=True
)


# ================================================================
# LOAD DATA
# ================================================================

print("\nMAIVF-HIS Validation Framework")
print("=" * 70)

print(f"Loading dataset: {args.data}")

df = pd.read_excel(args.data)

df[target] = pd.to_numeric(
    df[target],
    errors="coerce"
)

df = df.dropna(
    subset=[target]
)

df = df[
    df[target] > 0
].reset_index(drop=True)


print(
    f"Records available for validation: {len(df):,}"
)


# ================================================================
# DETECT DIAGNOSIS / PROCEDURE COLUMNS
# ================================================================

diag = [
    c for c in df.columns
    if (
        c == "Primary_Diagnosis"
        or c.lower().startswith("secondary_diag")
    )
]

proc = [
    c for c in df.columns
    if (
        c == "Primary_Procedure"
        or c.lower().startswith("secondary_proc")
    )
]

features = diag + proc


if not features:
    raise ValueError(
        "No diagnosis/procedure columns detected."
    )


print(
    f"Diagnosis columns detected: {len(diag)}"
)

print(
    f"Procedure columns detected: {len(proc)}"
)


# ================================================================
# FEATURE HASHING
# ================================================================

hasher = FeatureHasher(
    n_features=cfg["model"]["hashing_features"],
    input_type="string",
    alternate_sign=False
)


def transform(data):

    rows = []

    for _, row in data[features].iterrows():

        tokens = []

        for c in features:

            if pd.notna(row[c]):

                tokens.append(
                    f"{c}={str(row[c]).strip()}"
                )

        rows.append(tokens)

    return hasher.transform(rows)


# ================================================================
# FIXED TRAIN / TEST PARTITION
# ================================================================

rng = np.random.default_rng(seed)

idx = np.arange(
    len(df)
)

rng.shuffle(idx)

split = int(
    len(idx)
    * cfg["validation"]["train_fraction"]
)

tr0 = idx[:split]
te0 = idx[split:]


def fit_eval(data):

    """
    The original D0 train/test indices are retained for all
    degradation conditions.

    D4 and D9 append synthetic duplicate records. These appended
    records are intentionally excluded from model evaluation so
    that changes in predictive performance are not caused by
    altered partition size.
    """

    tr = data.iloc[tr0]
    te = data.iloc[te0]

    Xtr = transform(tr)
    Xte = transform(te)

    ytr = (
        tr[target]
        .astype(float)
        .values
    )

    yte = (
        te[target]
        .astype(float)
        .values
    )

    model = Ridge(
        alpha=cfg["model"]["ridge_alpha"]
    )

    model.fit(
        Xtr,
        ytr
    )

    pred = model.predict(
        Xte
    )

    metrics = regression_metrics(
        yte,
        pred
    )

    return (
        model,
        metrics,
        yte,
        pred
    )


# ================================================================
# REFERENCE CONDITION D0
# ================================================================

print("\nRunning reference model (D0)...")

_, ref, _, _ = fit_eval(df)


# ================================================================
# CONTROLLED DEGRADATION CONDITIONS
# ================================================================

conditions = generate_degradation_conditions(
    df,
    features,
    diag,
    proc,
    seed
)


# ================================================================
# MAPDS
# ================================================================

def mapds(scores):

    applicable = [
        value
        for value in scores.values()
        if pd.notna(value)
    ]

    if not applicable:
        return np.nan

    return float(
        np.mean(applicable)
    )


# ================================================================
# D0, D1, D2, D3, D4, D5, D9
# ================================================================

results = []


for cond in [
    "D0",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D9"
]:

    print(
        f"Running {cond}..."
    )

    data = conditions[cond]


    # ------------------------------------------------------------
    # Controlled duplication
    #
    # D4 = 5% controlled duplication
    # D9 = combined condition including 5% duplication
    # ------------------------------------------------------------

    controlled_duplicate_fraction = (
        0.05
        if cond in ["D4", "D9"]
        else None
    )


    # ------------------------------------------------------------
    # G1 — DATA QUALITY
    # ------------------------------------------------------------

    q = evaluate_data_quality(
        data=data,
        reference=df,
        feature_columns=features,
        diagnosis_columns=diag,
        procedure_columns=proc,
        target_column=target,
        id_column=None,
        controlled_duplicate_fraction=
            controlled_duplicate_fraction
    )


    # ------------------------------------------------------------
    # G3 — PREDICTIVE PERFORMANCE
    # ------------------------------------------------------------

    _, met, _, _ = fit_eval(
        data
    )

    g3 = normalized_regression_score(
        ref,
        met
    )


    # ------------------------------------------------------------
    # G4 — TEMPORAL ROBUSTNESS
    # ------------------------------------------------------------

    g4_result = (
        evaluate_temporal_robustness()
    )

    g4 = g4_result["G4"]


    # ------------------------------------------------------------
    # G5 — CALIBRATION
    # ------------------------------------------------------------

    g5_result = (
        evaluate_calibration(
            "regression"
        )
    )

    g5 = g5_result["G5"]


    # ------------------------------------------------------------
    # G6 — UNCERTAINTY / EXPLAINABILITY
    # ------------------------------------------------------------

    g6_result = (
        evaluate_uncertainty_explainability()
    )

    g6 = g6_result["G6"]


    # ------------------------------------------------------------
    # G7 — HIS INTEGRATION
    # ------------------------------------------------------------

    g7_result = (
        evaluate_his_integration()
    )

    g7 = g7_result["G7"]


    # ------------------------------------------------------------
    # MAPDS
    # ------------------------------------------------------------

    score = mapds(
        {
            "G1": q["G1"],
            "G3": g3,
            "G4": g4,
            "G5": g5,
            "G6": g6,
            "G7": g7
        }
    )


    results.append(
        {
            "Condition": cond,

            **q,
            **met,

            "G2_Critical": False,

            "G3": g3,
            "G4": g4,
            "G5": g5,
            "G6": g6,
            "G7": g7,

            "MAPDS": score,

            "Final_Classification":
                "Not assigned — empirical threshold required"
        }
    )


# ================================================================
# D6 — TARGET LEAKAGE POSITIVE CONTROL
# ================================================================

print("Running D6 — target leakage positive control...")


tr = df.iloc[tr0]
te = df.iloc[te0]


Xtr = sparse.hstack(
    [
        transform(tr),

        tr[target]
        .astype(float)
        .values
        .reshape(-1, 1)
    ],
    format="csr"
)


Xte = sparse.hstack(
    [
        transform(te),

        te[target]
        .astype(float)
        .values
        .reshape(-1, 1)
    ],
    format="csr"
)


model = Ridge(
    alpha=cfg["model"]["ridge_alpha"]
)

model.fit(
    Xtr,
    tr[target]
)


pred = model.predict(
    Xte
)


met = regression_metrics(
    te[target],
    pred
)


q = evaluate_data_quality(
    data=df,
    reference=df,
    feature_columns=features,
    diagnosis_columns=diag,
    procedure_columns=proc,
    target_column=target,
    id_column=None,
    controlled_duplicate_fraction=None
)


g3 = normalized_regression_score(
    ref,
    met
)


results.append(
    {
        "Condition": "D6",

        **q,
        **met,

        "G2_Critical": True,

        "G3": g3,

        "G4": np.nan,
        "G5": np.nan,
        "G6": np.nan,
        "G7": np.nan,

        "MAPDS": mapds(
            {
                "G1": q["G1"],
                "G3": g3
            }
        ),

        # Non-compensatory critical-failure rule.
        "Final_Classification":
            "Not Ready"
    }
)


# ================================================================
# D7 — TEMPORAL DISTRIBUTION SHIFT
# ================================================================

print("Recording D7...")

results.append(
    {
        "Condition": "D7",

        "G2_Critical": False,

        "G4": np.nan,

        "MAPDS": np.nan,

        "Final_Classification":
            "Not Evaluated — temporal variable unavailable"
    }
)


# ================================================================
# D8 — CALIBRATION DISTORTION
# ================================================================

print("Recording D8...")

results.append(
    {
        "Condition": "D8",

        "G2_Critical": False,

        "G5": np.nan,

        "MAPDS": np.nan,

        "Final_Classification":
            (
                "Not Applicable — probability calibration "
                "not evaluated for the continuous regression "
                "reference output"
            )
    }
)


# ================================================================
# CREATE RESULTS DATAFRAME
# ================================================================

r = pd.DataFrame(
    results
)


order = [
    f"D{i}"
    for i in range(10)
]


r["Condition"] = pd.Categorical(
    r["Condition"],
    categories=order,
    ordered=True
)


r = (
    r.sort_values("Condition")
    .reset_index(drop=True)
)


# ================================================================
# DELTA MAPDS RELATIVE TO D0
# ================================================================

d0 = (
    r.loc[
        r["Condition"] == "D0",
        "MAPDS"
    ]
    .iloc[0]
)


r["Delta_MAPDS"] = (
    r["MAPDS"]
    - d0
)


# ================================================================
# TABLE 6 — PROGRESSIVE MISSINGNESS
# ================================================================

t6 = r[
    r["Condition"].isin(
        [
            "D0",
            "D1",
            "D2",
            "D3"
        ]
    )
][
    [
        "Condition",
        "Completeness",
        "Validity",
        "Consistency",
        "Uniqueness",
        "Uniqueness_Status",
        "DQS",
        "G1"
    ]
].copy()


t6[
    "Induced_Missingness"
] = [
    "0%",
    "5%",
    "10%",
    "20%"
]


# ================================================================
# TABLE 7 — CONTROLLED DEGRADATION
# ================================================================

t7 = r.reindex(
    columns=[
        "Condition",

        "G1",
        "G1_Applicable_Components",

        "G2_Critical",

        "G3",
        "G4",
        "G5",
        "G6",
        "G7",

        "MAE",
        "RMSE",
        "R2",

        "MAPDS",
        "Delta_MAPDS",

        "Final_Classification"
    ]
)


# ================================================================
# SAVE CSV FILES
# ================================================================

r.to_csv(
    f"{outdir}/MAIVF_HIS_Full_Results.csv",
    index=False
)


t6.to_csv(
    f"{outdir}/Table_6_Progressive_Missingness.csv",
    index=False
)


t7.to_csv(
    f"{outdir}/Table_7_Controlled_Degradation.csv",
    index=False
)


# ================================================================
# SAVE EXCEL REPORT
# ================================================================

excel_path = (
    f"{outdir}/MAIVF_HIS_Validation_Report.xlsx"
)


with pd.ExcelWriter(
    excel_path,
    engine="openpyxl"
) as w:

    t6.to_excel(
        w,
        sheet_name="Table 6",
        index=False
    )

    t7.to_excel(
        w,
        sheet_name="Table 7",
        index=False
    )

    r.to_excel(
        w,
        sheet_name="Full Results",
        index=False
    )


# ================================================================
# FIGURE 3 — CONTROLLED DEGRADATION RESPONSE
# ================================================================

fig_data = r[
    r["MAPDS"].notna()
].copy()


fig_data[
    "Condition"
] = (
    fig_data["Condition"]
    .astype(str)
)


plt.figure(
    figsize=(9, 5.5)
)


plt.plot(
    fig_data["Condition"],
    fig_data["MAPDS"],
    marker="o",
    linewidth=1.8,
    markersize=6
)


plt.xlabel(
    "Controlled degradation condition"
)


plt.ylabel(
    "Medical AI Pre-Deployment Score (MAPDS)"
)


plt.title(
    "MAIVF-HIS Response to Controlled Degradation"
)


plt.ylim(
    0,
    105
)


plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.30
)


plt.tight_layout()


figure_path = (
    f"{outdir}/Figure_3_MAPDS_Controlled_Degradation.png"
)


plt.savefig(
    figure_path,
    dpi=600,
    bbox_inches="tight"
)


plt.close()


# ================================================================
# TERMINAL SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("MAIVF-HIS VALIDATION COMPLETED")
print("=" * 70)


print(
    f"\nExcel report:\n{excel_path}"
)


print(
    f"\nFigure 3:\n{figure_path}"
)


print(
    "\nD0 reference performance:"
)


print(
    f"MAE  = {ref['MAE']:.4f}"
)


print(
    f"RMSE = {ref['RMSE']:.4f}"
)


print(
    f"R2   = {ref['R2']:.4f}"
)


print(
    "\nMAPDS by condition:"
)


print(
    r[
        [
            "Condition",
            "G1",
            "G2_Critical",
            "G3",
            "MAPDS",
            "Delta_MAPDS",
            "Final_Classification"
        ]
    ].to_string(
        index=False
    )
)