# MAIVF-HIS

**Medical AI Validation Framework for Hospital Information Systems (MAIVF-HIS)** is a pre-deployment validation framework for assessing candidate medical AI systems beyond predictive performance.

## Validation gates

- G1 — Data Quality
- G2 — Data Leakage
- G3 — Predictive Performance
- G4 — Temporal Robustness
- G5 — Calibration
- G6 — Uncertainty & Explainability
- G7 — HIS Integration Readiness

Critical G2 or G7 failures are non-compensatory and result in a **Not Ready** decision irrespective of aggregate score.

## Repository structure

```text
MAIVF-HIS/
├── config/validation_config.yaml
├── data/README.md
├── gates/
│   ├── data_quality.py
│   ├── leakage.py
│   ├── performance.py
│   ├── temporal_robustness.py
│   ├── calibration.py
│   ├── uncertainty_explainability.py
│   └── his_integration.py
├── degradation/controlled_degradation.py
├── examples/example_validation_workflow.ipynb
├── outputs/validation_report/
├── maivf_his.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The hospital dataset is intentionally **not distributed in this public repository**. Put an authorized local copy in `data/` or another local path. The `.gitignore` prevents common tabular data formats in `data/` from being committed accidentally.

The reference implementation expects `LOS`, diagnosis columns (`Primary_Diagnosis`, `Secondary_diag*`) and procedure columns (`Primary_Procedure`, `Secondary_proc*`). `INACBG` and `DESCRIPTION`, if present, are excluded from the predictors.

## Run

```bash
python maivf_his.py --data "data/dataset_cp_fix_2(20261007-064753).xlsx"
```

Generated reports are written to `outputs/validation_report/`.

## Controlled degradation

D0 is the reference condition. D1–D3 induce 5%, 10%, and 20% missingness only in originally observed predictor cells; D4 introduces duplication; D5 coding noise; D6 target leakage as a positive control; D7 is not evaluated without a temporal variable; D8 is not applicable to this regression demonstration; and D9 combines missingness, coding noise, and duplication.

## Important scope note

The included LOS regression model is a **reference implementation for demonstrating MAIVF-HIS**, not the methodological contribution itself. Gate applicability depends on the task and available evidence. Not Evaluated and Not Applicable gates are not treated as passed and are not assigned artificial scores.
