def evaluate_leakage(feature_columns,target_column,force_critical=False):
    t=target_column.lower(); suspicious=[c for c in feature_columns if c.lower()==t or t in c.lower()]
    critical=bool(suspicious) or force_critical
    return {"G2_Critical":critical,"Suspicious_Features":suspicious,"Status":"Critical leakage detected" if critical else "No critical leakage detected"}
