import numpy as np
import pandas as pd

def induce_missingness(data,features,fraction,seed):
    out=data.copy(); rng=np.random.default_rng(seed); pos=[]
    for c in features:
        pos.extend((r,c) for r in np.where(out[c].notna().values)[0])
    n=int(len(pos)*fraction)
    if n:
        for i in rng.choice(len(pos),size=n,replace=False): out.at[pos[i][0],pos[i][1]]=np.nan
    return out

def induce_duplication(data,fraction,seed):
    n=int(len(data)*fraction); dup=data.sample(n=n,replace=False,random_state=seed)
    return pd.concat([data,dup],ignore_index=True)

def induce_coding_noise(
    data,
    diagnosis,
    procedure,
    fraction=0.05,
    random_state=42
):
    """
    Introduce controlled coding noise into originally observed
    diagnosis and procedure codes.

    Columns are converted to object dtype before injecting invalid
    string codes because Excel may load code columns as numeric
    (e.g., float64).
    """

    out = data.copy()

    rng = np.random.default_rng(random_state)

    code_columns = diagnosis + procedure

    # Convert code columns to object dtype so that synthetic
    # invalid string codes can be inserted safely.
    for col in code_columns:
        if col in out.columns:
            out[col] = out[col].astype("object")

    positions = []

    # Only originally observed code values are eligible for
    # controlled coding-noise injection.
    for col in code_columns:

        if col not in out.columns:
            continue

        rows = np.where(
            out[col].notna().to_numpy()
        )[0]

        for row in rows:
            positions.append((row, col))

    if len(positions) == 0:
        return out

    n_noise = int(
        len(positions) * fraction
    )

    if n_noise == 0:
        return out

    selected = rng.choice(
        len(positions),
        size=n_noise,
        replace=False
    )

    for index in selected:

        row, col = positions[index]

        if col in diagnosis:
            out.at[row, col] = "INVALID_ICD10"
        else:
            out.at[row, col] = "INVALID_ICD9"

    return out

def generate_degradation_conditions(data,features,diagnosis,procedure,seed=42):
    d={"D0":data.copy()}
    d["D1"]=induce_missingness(data,features,.05,seed+1)
    d["D2"]=induce_missingness(data,features,.10,seed+2)
    d["D3"]=induce_missingness(data,features,.20,seed+3)
    d["D4"]=induce_duplication(data,.05,seed+4)
    d["D5"]=induce_coding_noise(data,diagnosis,procedure,.05,seed+5)
    d["D6"]=data.copy(); d["D7"]=None; d["D8"]=None
    x=induce_missingness(data,features,.10,seed+9); x=induce_coding_noise(x,diagnosis,procedure,.05,seed+10); x=induce_duplication(x,.05,seed+11); d["D9"]=x
    return d
