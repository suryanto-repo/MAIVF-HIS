import re
import numpy as np
import pandas as pd


# ================================================================
# ICD FORMAT RULES
# ================================================================

ICD10_PATTERN = re.compile(
    r"^[A-Z][0-9A-Z]{1,3}(\.[0-9A-Z]{1,4})?$"
)

ICD9_PATTERN = re.compile(
    r"^[0-9]{1,2}(\.[0-9]{1,4})?$"
)


def _valid_icd10(value):

    if pd.isna(value):
        return True

    value = str(value).strip().upper()

    return bool(
        ICD10_PATTERN.match(value)
    )


def _valid_icd9(value):

    if pd.isna(value):
        return True

    value = str(value).strip().upper()

    return bool(
        ICD9_PATTERN.match(value)
    )


# ================================================================
# COMPLETENESS
# ================================================================

def completeness(
    data,
    reference,
    feature_columns
):
    """
    Completeness is evaluated relative to predictor values that were
    observed in the D0 reference condition.

    Structural absence of secondary diagnoses or procedures is not
    treated as a missing-data error.
    """

    n = min(
        len(data),
        len(reference)
    )

    current = (
        data.iloc[:n][feature_columns]
        .reset_index(drop=True)
    )

    ref = (
        reference.iloc[:n][feature_columns]
        .reset_index(drop=True)
    )

    expected_mask = ref.notna()

    n_expected = int(
        expected_mask.sum().sum()
    )

    if n_expected == 0:
        return np.nan

    available_mask = (
        current.notna()
        & expected_mask
    )

    n_available = int(
        available_mask.sum().sum()
    )

    return (
        n_available
        / n_expected
    )


# ================================================================
# VALIDITY
# ================================================================

def validity(
    data,
    diagnosis_columns,
    procedure_columns
):

    valid = 0
    total = 0

    for col in diagnosis_columns:

        if col not in data.columns:
            continue

        for value in data[col].dropna():

            total += 1

            if _valid_icd10(value):
                valid += 1


    for col in procedure_columns:

        if col not in data.columns:
            continue

        for value in data[col].dropna():

            total += 1

            if _valid_icd9(value):
                valid += 1


    if total == 0:
        return np.nan

    return (
        valid
        / total
    )


# ================================================================
# CONSISTENCY
# ================================================================

def consistency(
    data,
    diagnosis_columns,
    procedure_columns
):
    """
    Demonstration consistency rules:

    1. Secondary diagnosis should not exist without a primary
       diagnosis.

    2. Secondary procedure should not exist without a primary
       procedure.

    These rules can be extended for other datasets.
    """

    total_checks = 0
    passed_checks = 0


    # ------------------------------------------------------------
    # Diagnosis consistency
    # ------------------------------------------------------------

    if "Primary_Diagnosis" in data.columns:

        secondary_diag = [
            c for c in diagnosis_columns
            if (
                c != "Primary_Diagnosis"
                and c in data.columns
            )
        ]

        if secondary_diag:

            has_secondary = (
                data[secondary_diag]
                .notna()
                .any(axis=1)
            )

            primary_available = (
                data[
                    "Primary_Diagnosis"
                ]
                .notna()
            )

            applicable = (
                has_secondary
            )

            total_checks += int(
                applicable.sum()
            )

            passed_checks += int(
                (
                    applicable
                    & primary_available
                )
                .sum()
            )


    # ------------------------------------------------------------
    # Procedure consistency
    # ------------------------------------------------------------

    if "Primary_Procedure" in data.columns:

        secondary_proc = [
            c for c in procedure_columns
            if (
                c != "Primary_Procedure"
                and c in data.columns
            )
        ]

        if secondary_proc:

            has_secondary = (
                data[secondary_proc]
                .notna()
                .any(axis=1)
            )

            primary_available = (
                data[
                    "Primary_Procedure"
                ]
                .notna()
            )

            applicable = (
                has_secondary
            )

            total_checks += int(
                applicable.sum()
            )

            passed_checks += int(
                (
                    applicable
                    & primary_available
                )
                .sum()
            )


    if total_checks == 0:
        return 1.0

    return (
        passed_checks
        / total_checks
    )


# ================================================================
# UNIQUENESS
# ================================================================

def uniqueness(
    data,
    id_column=None,
    controlled_duplicate_fraction=None
):
    """
    Uniqueness must not be inferred from identical clinical
    characteristics because two legitimate hospital episodes may
    contain identical diagnosis/procedure combinations.

    If a true encounter identifier is available, uniqueness is
    calculated from that identifier.

    For a controlled duplication experiment, the known synthetic
    duplication fraction may be used.

    Otherwise uniqueness is recorded as Not Evaluated (NaN).
    """

    # ------------------------------------------------------------
    # True encounter identifier available
    # ------------------------------------------------------------

    if (
        id_column is not None
        and id_column in data.columns
    ):

        valid_ids = (
            data[id_column]
            .dropna()
        )

        if len(valid_ids) == 0:
            return np.nan

        duplicates = (
            valid_ids
            .duplicated()
            .sum()
        )

        return (
            1
            -
            duplicates
            / len(valid_ids)
        )


    # ------------------------------------------------------------
    # Controlled D4 duplication
    # ------------------------------------------------------------

    if controlled_duplicate_fraction is not None:

        return max(
            0.0,
            1.0
            -
            float(
                controlled_duplicate_fraction
            )
        )


    # ------------------------------------------------------------
    # No reliable episode identifier
    # ------------------------------------------------------------

    return np.nan


# ================================================================
# DATA QUALITY SCORE
# ================================================================

def evaluate_data_quality(
    data,
    reference,
    feature_columns,
    diagnosis_columns,
    procedure_columns,
    target_column=None,
    id_column=None,
    controlled_duplicate_fraction=None
):

    C = completeness(
        data,
        reference,
        feature_columns
    )

    V = validity(
        data,
        diagnosis_columns,
        procedure_columns
    )

    K = consistency(
        data,
        diagnosis_columns,
        procedure_columns
    )

    U = uniqueness(
        data,
        id_column=id_column,
        controlled_duplicate_fraction=
            controlled_duplicate_fraction
    )


    # ------------------------------------------------------------
    # Only applicable components contribute to DQS.
    # ------------------------------------------------------------

    components = {
        "Completeness": C,
        "Validity": V,
        "Consistency": K,
        "Uniqueness": U
    }


    applicable = {
        key: value
        for key, value
        in components.items()
        if pd.notna(value)
    }


    if len(applicable) == 0:

        dqs = np.nan

    else:

        # Equal-weight reference implementation with automatic
        # renormalization over applicable components.

        dqs = (
            np.mean(
                list(
                    applicable.values()
                )
            )
            * 100
        )


    return {

        "Completeness":
            C * 100
            if pd.notna(C)
            else np.nan,

        "Validity":
            V * 100
            if pd.notna(V)
            else np.nan,

        "Consistency":
            K * 100
            if pd.notna(K)
            else np.nan,

        "Uniqueness":
            U * 100
            if pd.notna(U)
            else np.nan,

        "Uniqueness_Status":
            (
                "Evaluated"
                if pd.notna(U)
                else
                "Not Evaluated"
            ),

        "DQS":
            dqs,

        "G1":
            dqs,

        "G1_Applicable_Components":
            ", ".join(
                applicable.keys()
            )
    }