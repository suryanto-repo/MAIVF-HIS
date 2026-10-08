import numpy as np

def evaluate_uncertainty_explainability(uncertainty_values=None, explanation_stability=None):
    if uncertainty_values is None and explanation_stability is None:
        return {"G6":np.nan,"Status":"Not Evaluated","Reason":"No predefined quantitative uncertainty/explainability evidence supplied."}
    scores=[]
    if uncertainty_values is not None: scores.append(np.clip(100*(1-np.mean(uncertainty_values)),0,100))
    if explanation_stability is not None: scores.append(np.clip(100*explanation_stability,0,100))
    return {"G6":float(np.mean(scores)),"Status":"Evaluated"}
