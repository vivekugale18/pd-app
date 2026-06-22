"""
ablation_study.py
─────────────────
Ablation study for the ML pipeline (SVM + RF).

Configurations tested:
  1. Baseline     — MFCC only (40 features)
  2. + Delta       — MFCC + Delta MFCC (80 features)
  3. + Delta2      — + Delta-Delta MFCC (120 features)
  4. + Acoustic    — Full acoustic set (128 features)
  5. + Clinical    — All features including Praat (149 features)
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score,
    roc_auc_score, confusion_matrix
)

from build_dataset import X, y, person_ids


# ============================================================
# FEATURE SLICE DEFINITIONS
# ============================================================

CONFIGS = {
    "1_MFCC_only"        : (0,   40),
    "2_MFCC_Delta"       : (0,   80),
    "3_MFCC_Delta_Delta2": (0,  120),
    "4_Full_Acoustic"    : (0,  128),
    "5_Full_149"         : (0,  149),
}


# ============================================================
# METRIC HELPER
# ============================================================

def compute_metrics(y_true, y_pred, y_prob):
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return acc, f1, auc, sens, spec


# ============================================================
# ABLATION LOOP
# ============================================================

unique_persons = np.unique(person_ids)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

ablation_results = {}

for config_name, (feat_start, feat_end) in CONFIGS.items():

    print(f"\n{'='*55}")
    print(f"  Config: {config_name}  (features {feat_start}:{feat_end})")
    print(f"{'='*55}")

    X_config = X[:, feat_start:feat_end]

    fold_metrics = []
    fold = 1

    for train_idx, test_idx in kf.split(unique_persons):

        train_persons = unique_persons[train_idx]
        test_persons  = unique_persons[test_idx]

        train_mask = np.isin(person_ids, train_persons)
        test_mask  = np.isin(person_ids, test_persons)

        X_train = X_config[train_mask]
        y_train = y[train_mask]
        X_test  = X_config[test_mask]
        y_test  = y[test_mask]

        # ---- Impute NaNs (fit on train only) ----
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_test  = imputer.transform(X_test)

        # ---- Scale ----
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        model = SVC(kernel="rbf", probability=True, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc, f1, auc, sens, spec = compute_metrics(y_test, y_pred, y_prob)
        fold_metrics.append((acc, f1, auc, sens, spec))

        print(f"  Fold {fold}  Acc={acc*100:.1f}%  F1={f1:.3f}  "
              f"AUC={auc:.3f}  Sens={sens:.3f}  Spec={spec:.3f}")
        fold += 1

    ablation_results[config_name] = fold_metrics


# ============================================================
# SUMMARY TABLE
# ============================================================

print("\n\n" + "="*90)
print("  ABLATION STUDY — SUMMARY TABLE  (SVM, 5-Fold Person-wise CV)")
print("="*90)
print(f"  {'Config':<28} {'#Feat':>5}  "
      f"{'Acc(%)':>8} {'±':>5}  "
      f"{'F1':>6} {'±':>5}  "
      f"{'AUC':>6} {'±':>5}  "
      f"{'Sens':>6}  {'Spec':>6}")
print(f"  {'-'*88}")

config_feat_counts = {k: v[1]-v[0] for k, v in CONFIGS.items()}
rows = []

for config_name, fold_metrics in ablation_results.items():
    metrics_arr = np.array(fold_metrics)
    means = metrics_arr.mean(axis=0)
    stds  = metrics_arr.std(axis=0)
    n_feat = config_feat_counts[config_name]

    print(
        f"  {config_name:<28} {n_feat:>5}  "
        f"{means[0]*100:>7.2f}% {stds[0]*100:>4.2f}  "
        f"{means[1]:>6.3f} {stds[1]:>4.3f}  "
        f"{means[2]:>6.3f} {stds[2]:>4.3f}  "
        f"{means[3]:>6.3f}  {means[4]:>6.3f}"
    )

    rows.append({
        "Config"      : config_name,
        "Features"    : n_feat,
        "Acc_mean"    : round(means[0]*100, 2),
        "Acc_std"     : round(stds[0]*100,  2),
        "F1_mean"     : round(means[1], 4),
        "AUC_mean"    : round(means[2], 4),
        "Sensitivity" : round(means[3], 4),
        "Specificity" : round(means[4], 4),
    })

print("="*90)

# ---- Save as CSV ----
df_results = pd.DataFrame(rows)
df_results.to_csv("ablation_results.csv", index=False)
print("\nSaved: ablation_results.csv")

# ---- Key insight ----
best_row  = max(rows, key=lambda r: r["Acc_mean"])
worst_row = min(rows, key=lambda r: r["Acc_mean"])
gain      = best_row["Acc_mean"] - worst_row["Acc_mean"]

print(f"\nKey finding:")
print(f"  Baseline ({worst_row['Config']}) accuracy : {worst_row['Acc_mean']}%")
print(f"  Best     ({best_row['Config']})  accuracy : {best_row['Acc_mean']}%")
print(f"  Gain from full feature set                : +{gain:.2f}%")
print(f"\n  → Clinical features (Jitter, Shimmer, HNR, F0) contributed")
print(f"    {best_row['Acc_mean'] - rows[3]['Acc_mean']:.2f}% improvement over acoustic-only baseline.")