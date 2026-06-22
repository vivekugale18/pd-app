import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Import dataset
from build_dataset import X, y, person_ids


print("Feature matrix shape:", X.shape)
print("Labels shape         :", y.shape)
print("Unique persons       :", len(np.unique(person_ids)))
print("Class distribution   — PD:", np.sum(y==1), "  HC:", np.sum(y==0))


# ============================================================
# NaN DIAGNOSTICS — understand where NaNs are coming from
# ============================================================

nan_mask = np.isnan(X)
total_nans = nan_mask.sum()
nan_segments = nan_mask.any(axis=1).sum()
nan_features = np.where(nan_mask.any(axis=0))[0]

print(f"\nNaN Diagnostics:")
print(f"  Total NaN values    : {total_nans}")
print(f"  Segments with NaNs  : {nan_segments} / {X.shape[0]}")
print(f"  Features with NaNs  : {len(nan_features)} — indices: {nan_features[:20]}{'...' if len(nan_features)>20 else ''}")

# Drop segments that are entirely NaN (silent/corrupt audio)
all_nan_mask = nan_mask.all(axis=1)
n_dropped = all_nan_mask.sum()
if n_dropped > 0:
    print(f"  Dropping {n_dropped} fully-NaN segments (silent/corrupt)...")
    keep_mask = ~all_nan_mask
    X          = X[keep_mask]
    y          = y[keep_mask]
    person_ids = person_ids[keep_mask]
    print(f"  Remaining segments  : {X.shape[0]}")

print(f"  Remaining NaNs will be imputed with per-feature median.\n")


# ============================================================
# HELPER — compute all clinical metrics from predictions
# ============================================================

def compute_metrics(y_true, y_pred, y_prob):
    """
    Returns dict with accuracy, f1, auc, sensitivity, specificity.
    y_prob  : probability scores for class 1 (PD)
    """
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob)

    cm             = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn)   # PD recall — most critical in clinical setting
    specificity = tn / (tn + fp)   # Healthy recall

    return {
        "accuracy"   : acc,
        "f1"         : f1,
        "auc"        : auc,
        "sensitivity": sensitivity,
        "specificity": specificity
    }


# ============================================================
# STORAGE — segment-level and person-level for both models
# ============================================================

results = {
    "SVM": {
        "seg_acc": [], "seg_f1": [], "seg_auc": [],
        "seg_sens": [], "seg_spec": [],
        "person_acc": [], "person_f1": [],
        "person_sens": [], "person_spec": []
    },
    "RandomForest": {
        "seg_acc": [], "seg_f1": [], "seg_auc": [],
        "seg_sens": [], "seg_spec": [],
        "person_acc": [], "person_f1": [],
        "person_sens": [], "person_spec": []
    }
}


# ============================================================
# 5-FOLD PERSON-WISE CROSS VALIDATION
# ============================================================

unique_persons = np.unique(person_ids)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold = 1

for train_index, test_index in kf.split(unique_persons):

    print(f"\n{'='*40}")
    print(f"  FOLD {fold}")
    print(f"{'='*40}")

    train_persons = unique_persons[train_index]
    test_persons  = unique_persons[test_index]

    train_mask = np.isin(person_ids, train_persons)
    test_mask  = np.isin(person_ids, test_persons)

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test  = X[test_mask]
    y_test  = y[test_mask]

    test_person_ids = person_ids[test_mask]

    # ---- Step 1: Impute NaNs (fit on train only to prevent leakage) ----
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test  = imputer.transform(X_test)

    # ---- Step 2: Scale (fit on train only) ----
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ---- Models ----
    models = {
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
    }

    for model_name, model in models.items():

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]   # probability for PD class

        # ---- Segment-level metrics ----
        seg_m = compute_metrics(y_test, y_pred, y_prob)

        print(f"\n  {model_name} — Segment Level")
        print(f"    Accuracy    : {seg_m['accuracy']*100:.2f}%")
        print(f"    F1 Score    : {seg_m['f1']:.4f}")
        print(f"    AUC-ROC     : {seg_m['auc']:.4f}")
        print(f"    Sensitivity : {seg_m['sensitivity']:.4f}")
        print(f"    Specificity : {seg_m['specificity']:.4f}")

        # ---- Person-level majority voting ----
        df = pd.DataFrame({
            "person"    : test_person_ids,
            "true_label": y_test,
            "pred_label": y_pred,
            "prob_pd"   : y_prob
        })

        # Majority vote on label; mean probability for AUC
        person_results = df.groupby("person").agg(
            true_label=("true_label", "first"),
            voted_label=("pred_label", lambda x: np.bincount(x).argmax()),
            mean_prob=("prob_pd", "mean")
        ).reset_index()

        p_acc  = accuracy_score(person_results["true_label"], person_results["voted_label"])
        p_f1   = f1_score(person_results["true_label"], person_results["voted_label"])
        p_auc  = roc_auc_score(person_results["true_label"], person_results["mean_prob"])

        p_cm             = confusion_matrix(person_results["true_label"], person_results["voted_label"])
        p_tn, p_fp, p_fn, p_tp = p_cm.ravel()
        p_sens = p_tp / (p_tp + p_fn)
        p_spec = p_tn / (p_tn + p_fp)

        print(f"\n  {model_name} — Person Level (Majority Vote)")
        print(f"    Accuracy    : {p_acc*100:.2f}%")
        print(f"    F1 Score    : {p_f1:.4f}")
        print(f"    AUC-ROC     : {p_auc:.4f}")
        print(f"    Sensitivity : {p_sens:.4f}")
        print(f"    Specificity : {p_spec:.4f}")

        # Store
        r = results[model_name]
        r["seg_acc"].append(seg_m["accuracy"])
        r["seg_f1"].append(seg_m["f1"])
        r["seg_auc"].append(seg_m["auc"])
        r["seg_sens"].append(seg_m["sensitivity"])
        r["seg_spec"].append(seg_m["specificity"])
        r["person_acc"].append(p_acc)
        r["person_f1"].append(p_f1)
        r["person_sens"].append(p_sens)
        r["person_spec"].append(p_spec)

    fold += 1


# ============================================================
# FINAL SUMMARY TABLE
# ============================================================

def print_summary(model_name, r):
    print(f"\n{'='*50}")
    print(f"  {model_name}  —  5-Fold CV Summary")
    print(f"{'='*50}")
    print(f"  {'Metric':<20} {'Seg Mean':>10} {'Seg Std':>10} {'Person Mean':>12} {'Person Std':>10}")
    print(f"  {'-'*64}")

    metrics = [
        ("Accuracy",    "seg_acc",  "person_acc"),
        ("F1 Score",    "seg_f1",   "person_f1"),
        ("Sensitivity", "seg_sens", "person_sens"),
        ("Specificity", "seg_spec", "person_spec"),
    ]

    for label, sk, pk in metrics:
        print(
            f"  {label:<20}"
            f" {np.mean(r[sk])*100:>9.2f}%"
            f" {np.std(r[sk])*100:>9.2f}%"
            f" {np.mean(r[pk])*100:>11.2f}%"
            f" {np.std(r[pk])*100:>9.2f}%"
        )

    # AUC only at segment level (person-level AUC also available)
    print(
        f"  {'AUC-ROC':<20}"
        f" {np.mean(r['seg_auc']):>10.4f}"
        f" {np.std(r['seg_auc']):>10.4f}"
    )


print_summary("SVM", results["SVM"])
print_summary("Random Forest", results["RandomForest"])


# ============================================================
# TRAIN FINAL MODELS ON FULL DATA & SAVE
# ============================================================

print("\n\nTraining final models on full dataset...")

# Step 1: Impute on full data
imputer_final = SimpleImputer(strategy="median")
X_imputed     = imputer_final.fit_transform(X)

# Step 2: Scale
scaler_final = StandardScaler()
X_scaled     = scaler_final.fit_transform(X_imputed)

# SVM
final_svm = SVC(kernel='rbf', probability=True, random_state=42)
final_svm.fit(X_scaled, y)

# Random Forest
final_rf = RandomForestClassifier(n_estimators=200, random_state=42)
final_rf.fit(X_scaled, y)

# Save all three — imputer MUST be applied before scaler at inference time
joblib.dump(imputer_final, "imputer.pkl")
joblib.dump(scaler_final,  "scaler.pkl")
joblib.dump(final_svm,     "svm_model.pkl")
joblib.dump(final_rf,      "rf_model.pkl")

print("Saved: imputer.pkl")
print("Saved: scaler.pkl")
print("Saved: svm_model.pkl")
print("Saved: rf_model.pkl")
print("\nDone.")