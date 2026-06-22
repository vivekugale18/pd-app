"""
shap_analysis.py
────────────────
Generates SHAP explanations for the trained Random Forest model.

Install:
    pip install shap

Output files saved:
    shap_values.npy          — raw SHAP values array (load in app.py)
    shap_feature_names.npy   — feature name strings
    shap_summary.png         — beeswarm plot (best for reports)
    shap_bar.png             — mean importance bar chart
    shap_waterfall_pd.png    — single PD patient explanation
    shap_waterfall_hc.png    — single Healthy patient explanation
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import shap

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from build_dataset import X, y, person_ids


# ============================================================
# FEATURE NAMES
# ============================================================

def build_feature_names():
    names = []
    for i in range(1, 21): names.append(f"MFCC_{i}_mean")
    for i in range(1, 21): names.append(f"MFCC_{i}_std")
    for i in range(1, 21): names.append(f"Delta_MFCC_{i}_mean")
    for i in range(1, 21): names.append(f"Delta_MFCC_{i}_std")
    for i in range(1, 21): names.append(f"Delta2_MFCC_{i}_mean")
    for i in range(1, 21): names.append(f"Delta2_MFCC_{i}_std")
    names += [
        "Pitch_mean", "Pitch_std",
        "ZCR_mean",   "ZCR_std",
        "Spectral_Centroid_mean", "Spectral_Centroid_std",
        "RMS_mean",   "RMS_std",
        "F0_mean", "F0_std", "F0_min", "F0_max",
        "Jitter_local", "Jitter_RAP", "Jitter_PPQ5", "Jitter_DDP",
        "Shimmer_local", "Shimmer_APQ3", "Shimmer_APQ5",
        "Shimmer_APQ11", "Shimmer_DDA",
        "HNR", "NHR",
        "Spectral_Flux_mean",      "Spectral_Flux_std",
        "Spectral_Rolloff_mean",   "Spectral_Rolloff_std",
        "Spectral_Bandwidth_mean", "Spectral_Bandwidth_std"
    ]
    assert len(names) == 149, f"Expected 149 names, got {len(names)}"
    return names

feature_names = build_feature_names()


# ============================================================
# LOAD MODEL & PREPROCESS  (impute → scale, same as train_ml.py)
# ============================================================

print("Loading trained Random Forest model...")
rf_model = joblib.load("rf_model.pkl")
imputer  = joblib.load("imputer.pkl")
scaler   = joblib.load("scaler.pkl")

X_imp    = imputer.transform(X)
X_scaled = scaler.transform(X_imp)

print(f"Dataset shape : {X_scaled.shape}")
print(f"Feature names : {len(feature_names)}")


# ============================================================
# COMPUTE SHAP VALUES  (modern API — returns Explanation object)
# ============================================================

print("\nComputing SHAP values (TreeExplainer)...")
print("This takes ~30–60 seconds for large datasets...\n")

explainer  = shap.TreeExplainer(rf_model)
shap_exp   = explainer(X_scaled)   # returns Explanation object, shape (n_samples, n_features, n_classes)

print(f"Raw SHAP output shape: {shap_exp.values.shape}")

# For binary RF: shape is (n_samples, n_features, 2)
# Index [:, :, 1] = contributions toward PD (class 1)
if shap_exp.values.ndim == 3:
    shap_pd       = shap_exp.values[:, :, 1]
    base_value_pd = shap_exp.base_values[0, 1] if shap_exp.base_values.ndim == 2 else shap_exp.base_values[1]
else:
    # Some versions return (n_samples, n_features) for binary
    shap_pd       = shap_exp.values
    base_value_pd = shap_exp.base_values[0] if hasattr(shap_exp.base_values, '__len__') else shap_exp.base_values

print(f"SHAP PD values shape : {shap_pd.shape}")
print(f"Base value (PD class): {base_value_pd:.4f}")
print("SHAP computation complete.\n")


# ============================================================
# SAVE RAW SHAP VALUES
# ============================================================

np.save("shap_values.npy",        shap_pd)
np.save("shap_feature_names.npy", np.array(feature_names))
print("Saved: shap_values.npy")
print("Saved: shap_feature_names.npy")


# ============================================================
# PLOT 1 — Beeswarm Summary Plot
# ============================================================

print("\nGenerating plots...")

# Build a clean Explanation object for PD class
shap_explanation = shap.Explanation(
    values        = shap_pd,
    base_values   = np.full(shap_pd.shape[0], base_value_pd),
    data          = X_scaled,
    feature_names = feature_names
)

plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_pd,
    X_scaled,
    feature_names=feature_names,
    max_display=20,
    show=False
)
plt.title("SHAP Summary — Top 20 Features (Parkinson's Class)", fontsize=14)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_summary.png")


# ============================================================
# PLOT 2 — Bar Chart
# ============================================================

plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_pd,
    X_scaled,
    feature_names=feature_names,
    plot_type="bar",
    max_display=20,
    show=False
)
plt.title("SHAP Feature Importance — Mean |SHAP| (Top 20)", fontsize=14)
plt.tight_layout()
plt.savefig("shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_bar.png")


# ============================================================
# PLOT 3 — Waterfall: Most confident PD prediction
# ============================================================

pd_indices  = np.where(y == 1)[0]
hc_indices  = np.where(y == 0)[0]

pd_probs    = rf_model.predict_proba(X_scaled[pd_indices])[:, 1]
best_pd_idx = pd_indices[np.argmax(pd_probs)]

plt.figure(figsize=(12, 7))
shap.waterfall_plot(
    shap.Explanation(
        values        = shap_pd[best_pd_idx],
        base_values   = base_value_pd,
        data          = X_scaled[best_pd_idx],
        feature_names = feature_names
    ),
    max_display=15,
    show=False
)
plt.title("SHAP Waterfall — Parkinson's Patient (highest confidence)", fontsize=13)
plt.tight_layout()
plt.savefig("shap_waterfall_pd.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_waterfall_pd.png")


# ============================================================
# PLOT 4 — Waterfall: Most confident HC prediction
# ============================================================

hc_probs    = rf_model.predict_proba(X_scaled[hc_indices])[:, 0]
best_hc_idx = hc_indices[np.argmax(hc_probs)]

plt.figure(figsize=(12, 7))
shap.waterfall_plot(
    shap.Explanation(
        values        = shap_pd[best_hc_idx],
        base_values   = base_value_pd,
        data          = X_scaled[best_hc_idx],
        feature_names = feature_names
    ),
    max_display=15,
    show=False
)
plt.title("SHAP Waterfall — Healthy Patient (highest confidence)", fontsize=13)
plt.tight_layout()
plt.savefig("shap_waterfall_hc.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_waterfall_hc.png")


# ============================================================
# PRINT TOP 15 FEATURES BY MEAN |SHAP|
# ============================================================

mean_shap   = np.abs(shap_pd).mean(axis=0)
top_indices = np.argsort(mean_shap)[::-1][:15]

print("\n" + "="*55)
print("  TOP 15 FEATURES BY MEAN |SHAP| VALUE")
print("="*55)
print(f"  {'Rank':<5} {'Feature':<35} {'Mean |SHAP|':>12}")
print(f"  {'-'*53}")
for rank, idx in enumerate(top_indices, 1):
    print(f"  {rank:<5} {feature_names[idx]:<35} {mean_shap[idx]:>12.5f}")

print("\nSHAP analysis complete.")
print("Files saved: shap_summary.png, shap_bar.png,")
print("             shap_waterfall_pd.png, shap_waterfall_hc.png")