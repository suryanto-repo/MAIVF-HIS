import numpy as np

def evaluate_calibration(task_type, y_true=None, probabilities=None):
    if task_type != "classification":
        return {"G5":np.nan,"Status":"Not Applicable","Reason":"Probability calibration is not applicable to the regression reference task."}
    return {"G5":np.nan,"Status":"Not Evaluated","Reason":"Provide a predefined probabilistic calibration protocol."}
