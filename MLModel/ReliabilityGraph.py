import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) Load calibration table
calib = pd.read_csv("feedback/calibration_table.csv")

# Guard for potential string bins: we only need mean_pred and win_rate
x = calib["mean_pred"].to_numpy()
y = calib["win_rate"].to_numpy()
n = calib["n"].to_numpy()

# 2) Simple (weighted) Expected Calibration Error (ECE)
#    ECE = sum_i (n_i / N) * |win_rate_i - mean_pred_i|
N = n.sum() if len(n) > 0 else 1
ece = float(np.sum((n / N) * np.abs(y - x)))

# 3) Plot reliability diagram
plt.figure(figsize=(6,6))
plt.scatter(x, y, s=np.clip(n*40, 20, 400), alpha=0.8, label="Bins")  # marker size ~ count
plt.plot([0,1], [0,1], linestyle="--", label="Perfect calibration")
# Optional smoothing/connection
order = np.argsort(x)
plt.plot(x[order], y[order], alpha=0.6, label="Calibration curve")

plt.title(f"Reliability Diagram (ECE={ece:.3f})")
plt.xlabel("Mean predicted probability")
plt.ylabel("Observed win rate")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 4) (Optional) Bar chart of bin counts
plt.figure(figsize=(6,3))
plt.bar(range(len(n)), n)
plt.title("Samples per bin")
plt.xlabel("Bin index")
plt.ylabel("Count")
plt.tight_layout()
plt.show()