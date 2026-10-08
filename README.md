# Medical AI Validation Framework for Hospital Information Systems (MAIVF-HIS)

![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.10--3.12-blue)
![License](https://img.shields.io/badge/license-MIT-yellow)

Reference implementation of the **Medical AI Validation Framework for Hospital Information Systems (MAIVF-HIS)**, a structured and reproducible pre-deployment validation framework for evaluating medical artificial intelligence before integration into hospital information systems (HIS).

MAIVF-HIS extends conventional predictive model evaluation by assessing deployment-relevant dimensions including data quality, methodological validity, temporal reliability, prediction trustworthiness, and operational readiness.

---

## Framework Overview

MAIVF-HIS evaluates a candidate medical AI system through seven complementary validation gates:

1. **G1 — Data Quality**
2. **G2 — Data Leakage**
3. **G3 — Predictive Performance**
4. **G4 — Temporal Robustness**
5. **G5 — Calibration**
6. **G6 — Uncertainty and Explainability**
7. **G7 — HIS Integration Readiness**

The framework separates gate-level evidence, critical-failure rules, and aggregate readiness scoring.

Confirmed critical failures in **G2 (Data Leakage)** or **G7 (HIS Integration Readiness)** are non-compensatory and result in a **Not Ready** classification regardless of aggregate predictive performance.

---

## Medical AI Pre-Deployment Score (MAPDS)

Applicable quantitative gate scores are normalized to a 0–100 scale and combined into the **Medical AI Pre-Deployment Score (MAPDS)**:

```text
MAPDS = Σ wi Gi
```

where the normalized weights satisfy:

```text
Σ wi = 1
```

When a quantitative gate is not applicable or cannot be evaluated, it is not assigned an artificial score. Instead, the corresponding weights are renormalized across the quantitatively applicable gates.

MAPDS is interpreted together with gate-level findings and critical-failure rules and should not be used as an independent indicator of deployment readiness.

---

## Critical-Failure Rules

MAIVF-HIS uses non-compensatory critical-failure rules.

```text
Critical G2 failure → NOT READY
Critical G7 failure → NOT READY
```

A high MAPDS cannot override a confirmed critical methodological or operational failure.

This design prevents apparently strong predictive performance from masking conditions such as target leakage or unsafe HIS integration behavior.

---

## Controlled Degradation

The reference implementation includes controlled degradation experiments for evaluating whether MAIVF-HIS responds appropriately to known data and methodological defects.

| Condition | Controlled degradation |
|---|---|
| D0 | Reference condition |
| D1 | 5% induced missingness |
| D2 | 10% induced missingness |
| D3 | 20% induced missingness |
| D4 | Controlled record duplication |
| D5 | Controlled coding noise |
| D6 | Introduced target leakage |
| D7 | Temporal distribution shift |
| D8 | Calibration distortion |
| D9 | Combined degradation |

D6 acts as a positive control for the critical-failure mechanism. Deliberately introduced target leakage may produce apparently excellent predictive performance, but MAIVF-HIS should identify the G2 critical failure and classify the model as **Not Ready**.

---

## Repository Structure

```text
MAIVF-HIS/
│
├── config/
│   └── validation_config.yaml
│
├── data/
│   └── README.md
│
├── degradation/
│   ├── __init__.py
│   └── controlled_degradation.py
│
├── examples/
│   └── example_validation_workflow.ipynb
│
├── gates/
│   ├── __init__.py
│   ├── data_quality.py
│   ├── leakage.py
│   ├── performance.py
│   ├── temporal_robustness.py
│   ├── calibration.py
│   ├── uncertainty_explainability.py
│   └── his_integration.py
│
├── outputs/
│   └── validation_report/
│
├── maivf_his.py
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/suryanto-repo/MAIVF-HIS.git
cd MAIVF-HIS
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Validation Workflow

The reference workflow can be executed using:

```bash
python maivf_his.py --data "data/your_dataset.xlsx"
```

The hospital dataset used in the methodological demonstration is **not distributed with this repository** because it contains non-public routinely collected health data.

Users should provide their own appropriately governed dataset and configure the validation workflow according to the intended AI task.

---

## Reference Demonstration

The current reference implementation demonstrates MAIVF-HIS using routinely collected hospital data containing:

- ICD-10 diagnosis codes
- ICD-9-CM procedure codes
- inpatient administrative and encounter information
- derived clinical-complexity features

A regression reference model is used to demonstrate the validation workflow and controlled degradation experiments.

The reference predictive model is used only to demonstrate the MAIVF-HIS methodology and should not be interpreted as the primary contribution of the repository.

---

## Validation Outputs

The workflow generates machine-readable validation outputs containing:

- component data-quality measures
- gate-level validation scores
- predictive-performance metrics
- critical-failure indicators
- MAPDS values
- changes relative to the D0 reference condition
- applicability status
- deployment-readiness assessment

Generated outputs can be retained in CSV and spreadsheet formats for independent inspection and audit.

---

## Reproducibility

Each validation run should retain sufficient information to reconstruct the evaluation, including:

- dataset identifier or version
- model identifier and version
- preprocessing configuration
- feature definitions
- random seed
- validation configuration
- controlled-degradation parameters
- gate-level outputs
- critical-failure findings
- MAPDS configuration
- software environment
- execution timestamp
- final deployment-readiness assessment

Controlled degradation experiments use fixed random seeds to support reproducibility.

---

## Data Availability and Privacy

No hospital or patient-level dataset is included in this repository.

The `data/` directory is configured to exclude spreadsheet, CSV, and other hospital datasets from Git version control.

Users are responsible for ensuring that datasets used with MAIVF-HIS comply with applicable ethical, institutional, privacy, and data-governance requirements.

---

## Scope

MAIVF-HIS is intended as a **pre-deployment technical and methodological screening framework**.

It does not replace:

- prospective clinical validation
- external validation
- regulatory assessment
- cybersecurity assessment
- clinical safety evaluation
- post-deployment monitoring

Deployment decisions should consider the intended use, clinical or operational risk, local requirements, and evidence beyond the MAIVF-HIS assessment.

---

## Citation

If you use MAIVF-HIS in your research, please cite the software as:

> Nugroho, S., Ginardi, R. V. H., & Purnama, I. K. E. (2026).  
> *Medical AI Validation Framework for Hospital Information Systems (MAIVF-HIS)* (Version 1.0.0) [Computer software].

A persistent DOI will be added after the MAIVF-HIS v1.0.0 release is archived in Zenodo.

---

## License

This project is distributed under the terms of the **MIT License**.

---

## Author

**Suryanto Nugroho**

Medical AI, Clinical Informatics, and Hospital Information Systems Research

---

> **Beyond predictive performance → deployment readiness for real-world use.**