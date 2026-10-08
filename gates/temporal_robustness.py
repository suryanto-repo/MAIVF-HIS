import numpy as np

def evaluate_temporal_robustness(date_column=None, reference_metric=None, temporal_metric=None, higher_is_better=False):
    if date_column is None:
        return {"G4":np.nan,"Status":"Not Evaluated","Reason":"Temporal variable unavailable"}
    if reference_metric in (None,0) or temporal_metric is None:
        return {"G4":np.nan,"Status":"Not Evaluated","Reason":"Temporal metrics unavailable"}
    degradation=((reference_metric-temporal_metric)/reference_metric*100) if higher_is_better else ((temporal_metric-reference_metric)/reference_metric*100)
    return {"G4":float(np.clip(100-max(degradation,0),0,100)),"Performance_Degradation":degradation,"Status":"Evaluated"}
