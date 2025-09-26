import numpy as np
import matplotlib.pyplot as plt

# From: TPR=Recall，FPR=FAR；3 Threshold：P95 / P99 / 17 m/s
roc = {
    "ConvLSTM": {
        "P95": (0.166, 0.679),
        "P99": (0.206, 0.560),
        "17 m/s": (0.000, 0.000),
    },
    "Informer": {
        "P95": (0.256, 0.650),
        "P99": (0.355, 0.575),
        "17 m/s": (0.565, 0.179),
    },
    "PatchTST": {
        "P95": (0.369, 0.693),
        "P99": (0.529, 0.520),
        "17 m/s": (0.626, 0.510),
    },
    "Patch-Informer": {
        "P95": (0.247, 0.638),
        "P99": (0.350, 0.561),
        "17 m/s": (0.573, 0.174),
    },
}

def approx_auc(points):
    pts = np.asarray(points, dtype=float)
    pts = pts[np.argsort(pts[:, 0])]
    if pts[0, 0] > 0:
        pts = np.vstack(([0.0, 0.0], pts))
    if pts[-1, 0] < 1:
        pts = np.vstack((pts, [1.0, 1.0]))
    return float(np.trapz(pts[:, 1], pts[:, 0]))

plt.figure(figsize=(7.2, 6))

for model, d in roc.items():
    # Original 3 points
    pts = np.array(list(d.values()), dtype=float)
    pts = pts[np.argsort(pts[:, 0])]

    # Draw line segment and add endpoints
    poly = pts.copy()
    if poly[0, 0] > 0: poly = np.vstack(([0.0, 0.0], poly))
    if poly[-1, 0] < 1: poly = np.vstack((poly, [1.0, 1.0]))

    auc = approx_auc(pts)
    plt.plot(poly[:, 0], poly[:, 1], marker='o', label=f"{model} (AUC≈{auc:.3f})")

    for name, (fx, ty) in d.items():
        plt.annotate(name, xy=(fx, ty), textcoords="offset points", xytext=(5, -10), fontsize=8)

x = np.linspace(0, 1, 50)
plt.plot(x, x, linestyle='--', linewidth=1, label="Random (AUC=0.5)")

plt.xlim(0, 1); plt.ylim(0, 1)
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Approximate ROC on Test set (discrete thresholds)")
plt.legend(loc="lower right", fontsize=8, frameon=False)
plt.grid(alpha=0.3, ls='--')
plt.tight_layout()

plt.savefig("approx_roc_test.png", dpi=200)
plt.show()