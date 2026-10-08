import numpy as np
REQUIRED_TESTS=["valid_input","missing_optional_input","missing_required_input","invalid_input","unknown_code","service_unavailable","api_failure","excessive_latency","prediction_logging","error_logging","model_version_traceability"]

def evaluate_his_integration(test_results=None):
    if test_results is None:
        return {"G7":np.nan,"Critical_G7":False,"Status":"Not Evaluated","Reason":"Operational HIS integration evidence unavailable."}
    vals=[test_results[k] for k in REQUIRED_TESTS if k in test_results]
    if not vals: return {"G7":np.nan,"Critical_G7":False,"Status":"Not Evaluated"}
    critical=(test_results.get("service_unavailable",True) is False or test_results.get("api_failure",True) is False)
    return {"G7":100*sum(bool(x) for x in vals)/len(vals),"Critical_G7":critical,"Status":"Critical failure" if critical else "Evaluated"}
