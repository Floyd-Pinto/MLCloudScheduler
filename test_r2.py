import numpy as np
from model.workload_generator import generate_spike
from sklearn.metrics import r2_score

s = generate_spike(steps=250, seed=42)
test_start = int(len(s)*0.7) + 20
test_actual = s[test_start:]
print(f"Variance of test_actual: {np.var(test_actual):.2f}")
