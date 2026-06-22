import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from build_dataset import X, y


# ============================================================
# FEATURE NAMES  (must match preprocessing.py order exactly)
# ============================================================

feature_names = (
    [f"MFCC_{i+1}_mean"       for i in range(20)] +
    [f"MFCC_{i+1}_std"        for i in range(20)] +
    [f"DeltaMFCC_{i+1}_mean"  for i in range(20)] +
    [f"DeltaMFCC_{i+1}_std"   for i in range(20)] +
    [f"Delta2MFCC_{i+1}_mean" for i in range(20)] +
    [f"Delta2MFCC_{i+1}_std"  for i in range(20)] +
    ["Pitch_mean", "Pitch_std"] +
    ["ZCR_mean",   "ZCR_std"] +
    ["SpectralCentroid_mean", "SpectralCentroid_std"] +
    ["RMS_mean",   "RMS_std"] +
    # Clinical (Praat) — 21 features
    ["F0_mean", "F0_std", "F0_min", "F0_max"] +
    ["Jitter_local", "Jitter_RAP", "Jitter_PPQ5", "Jitter_DDP"] +
    ["Shimmer_local", "Shimmer_APQ3", "Shimmer_APQ5",
     "Shimmer_APQ11", "Shimmer_DDA"] +
    ["HNR", "NHR"] +
    # Additional spectral — 6 features
    ["SpectralFlux_mean",      "SpectralFlux_std"] +
    ["SpectralRolloff_mean",   "SpectralRolloff_std"] +
    ["SpectralBandwidth_mean", "SpectralBandwidth_std"]
)

assert len(feature_names) == 149, f"Name count mismatch: {len(feature_names)}"


# ============================================================
# PREPROCESS  (impute → scale, same as train_ml.py)
# ============================================================

imputer  = SimpleImputer(strategy="median")
X_imp    = imputer.fit_transform(X)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)


# ============================================================
# TRAIN RANDOM FOREST ON FULL DATA
# ============================================================

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_scaled, y)

importances = model.feature_importances_
indices     = np.argsort(importances)[::-1]


# ============================================================
# PRINT TOP 20
# ============================================================

print("\nTop 20 Most Important Features:\n")
print(f"  {'Rank':<5} {'Feature':<30} {'Importance':>10}")
print(f"  {'-'*48}")

for rank, idx in enumerate(indices[:20], start=1):
    print(f"  {rank:<5} {feature_names[idx]:<30} {importances[idx]:>10.4f}")


# ============================================================
# PRINT BY FEATURE GROUP  (which group contributes most?)
# ============================================================

groups = {
    "MFCC (mean+std)"         : list(range(0,   40)),
    "Delta MFCC (mean+std)"   : list(range(40,  80)),
    "Delta2 MFCC (mean+std)"  : list(range(80,  120)),
    "Pitch"                   : list(range(120, 122)),
    "ZCR"                     : list(range(122, 124)),
    "Spectral Centroid"       : list(range(124, 126)),
    "RMS Energy"              : list(range(126, 128)),
    "F0 stats (Praat)"        : list(range(128, 132)),
    "Jitter (Praat)"          : list(range(132, 136)),
    "Shimmer (Praat)"         : list(range(136, 141)),
    "HNR / NHR (Praat)"       : list(range(141, 143)),
    "Spectral Flux"           : list(range(143, 145)),
    "Spectral Rolloff"        : list(range(145, 147)),
    "Spectral Bandwidth"      : list(range(147, 149)),
}

print("\n\nFeature Group Contributions:\n")
print(f"  {'Group':<30} {'Total Importance':>18} {'% of Total':>12}")
print(f"  {'-'*62}")

group_totals = {}
for group_name, idxs in groups.items():
    total = importances[idxs].sum()
    group_totals[group_name] = total

for group_name, total in sorted(group_totals.items(), key=lambda x: -x[1]):
    print(f"  {group_name:<30} {total:>18.4f} {total*100:>11.2f}%")


# ============================================================
# PLOT 1 — Top 20 features (bar chart)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Left: top 20 individual features
ax1 = axes[0]
top_n = 20
colors = ["#e74c3c" if importances[i] > 0.02 else "#3498db" for i in indices[:top_n]]
bars = ax1.barh(
    range(top_n),
    importances[indices[:top_n]],
    color=colors,
    edgecolor="white"
)
ax1.set_yticks(range(top_n))
ax1.set_yticklabels([feature_names[i] for i in indices[:top_n]], fontsize=9)
ax1.invert_yaxis()
ax1.set_xlabel("Importance Score")
ax1.set_title("Top 20 Most Important Features", fontweight="bold")
ax1.grid(axis="x", alpha=0.3)

# Right: group-level pie chart
ax2 = axes[1]
sorted_groups = sorted(group_totals.items(), key=lambda x: -x[1])
labels  = [g[0] for g in sorted_groups]
sizes   = [g[1] for g in sorted_groups]
explode = [0.05 if i == 0 else 0 for i in range(len(labels))]

wedges, texts, autotexts = ax2.pie(
    sizes,
    labels=None,
    autopct="%1.1f%%",
    explode=explode,
    startangle=140
)
ax2.legend(wedges, labels, loc="lower left", fontsize=8, bbox_to_anchor=(-0.1, -0.15))
ax2.set_title("Feature Group Contributions", fontweight="bold")

plt.suptitle("Random Forest Feature Importance — PD vs HC", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved: feature_importance.png")