# Data

Hospital-level source data are **not included** in this public repository.

For the demonstration workflow, place the local validation dataset in this directory and pass its path to `maivf_his.py` using `--data`.

Expected fields for the supplied reference implementation:
- `LOS` target variable
- `Primary_Diagnosis` and/or columns beginning with `Secondary_diag`
- `Primary_Procedure` and/or columns beginning with `Secondary_proc`

`INACBG` and `DESCRIPTION`, when present, are not used as predictors in the reference implementation.
