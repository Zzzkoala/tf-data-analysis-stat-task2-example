import pandas as pd
import numpy as np

from scipy.stats import norm


chat_id = 1308528894

def solution(p: float, x: np.array) -> tuple:
    n = len(x)
    alpha = 1 - p
    loc = x.mean()
    scale = np.sqrt(np.var(x)) / np.sqrt(len(x))
    return loc - scale * norm.ppf(1 - alpha / 2), \
           loc - scale * norm.ppf(alpha / 2)

# Test the solution function with a sample input
sample_p = 0.95
sample_distances = np.random.normal(0, 330, size=10)
interval = solution(sample_p, sample_distances)
#print("Confidence interval:", interval)
