import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score,
    roc_auc_score, classification_report,
    confusion_matrix
)

from federated_model import ParkinsonNN
from federated_data import X1, y1, X2, y2, X3, y3


# ============================================================
# STEP 1 — Per-Client Train/Test Split (BEFORE any scaling)
# ============================================================

def split_client_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test


print("Splitting each client into train / test...")

X1_train, X1_test, y1_train, y1_test = split_client_data(X1, y1)
X2_train, X2_test, y2_train, y2_test = split_client_data(X2, y2)
X3_train, X3_test, y3_train, y3_test = split_client_data(X3, y3)

print(f"Client 1 — Train: {X1_train.shape[0]}  Test: {X1_test.shape[0]}")
print(f"Client 2 — Train: {X2_train.shape[0]}  Test: {X2_test.shape[0]}")
print(f"Client 3 — Train: {X3_train.shape[0]}  Test: {X3_test.shape[0]}")


# ============================================================
# STEP 2 — Impute NaNs per client (fit on train only)
# NaNs exist in features 132-140 (Jitter/Shimmer from Praat)
# PyTorch won't raise a clear error for NaN inputs — model
# will silently diverge with NaN losses if not handled here.
# ============================================================

def impute_and_scale(X_train, X_test):
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test  = imputer.transform(X_test)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, imputer, scaler


X1_train_sc, X1_test_sc, imp1, scaler1 = impute_and_scale(X1_train, X1_test)
X2_train_sc, X2_test_sc, imp2, scaler2 = impute_and_scale(X2_train, X2_test)
X3_train_sc, X3_test_sc, imp3, scaler3 = impute_and_scale(X3_train, X3_test)

# Save imputers and scalers — both needed at inference time
joblib.dump(imp1,    "fl_imputer_client1.pkl")
joblib.dump(imp2,    "fl_imputer_client2.pkl")
joblib.dump(imp3,    "fl_imputer_client3.pkl")
joblib.dump(scaler1, "fl_scaler_client1.pkl")
joblib.dump(scaler2, "fl_scaler_client2.pkl")
joblib.dump(scaler3, "fl_scaler_client3.pkl")

print("\nClient imputers and scalers saved.")


# ============================================================
# STEP 3 — Local Training Function
# ============================================================

def train_local(model, X_scaled, y, epochs=5):
    """
    Trains model on pre-scaled, pre-imputed data.
    Returns updated state_dict.
    """
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss    = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    return model.state_dict()


# ============================================================
# STEP 4 — Weighted Federated Averaging (FedAvg)
# ============================================================

def federated_avg(weights_list, data_sizes):
    avg_weights = {}
    total = sum(data_sizes)

    for key in weights_list[0].keys():
        avg_weights[key] = sum(
            weights_list[i][key] * (data_sizes[i] / total)
            for i in range(len(weights_list))
        )

    return avg_weights


# ============================================================
# STEP 5 — Federated Training Loop
# ============================================================

global_model = ParkinsonNN()

ROUNDS     = 10
data_sizes = [len(X1_train), len(X2_train), len(X3_train)]

round_accuracies = []

print("\n" + "="*50)
print("FEDERATED TRAINING STARTED")
print("="*50)

for r in range(ROUNDS):

    print(f"\n===== Round {r+1} / {ROUNDS} =====")

    local_weights = []

    # Client 1
    model1 = ParkinsonNN()
    model1.load_state_dict(global_model.state_dict())
    w1 = train_local(model1, X1_train_sc, y1_train)
    local_weights.append(w1)

    # Client 2
    model2 = ParkinsonNN()
    model2.load_state_dict(global_model.state_dict())
    w2 = train_local(model2, X2_train_sc, y2_train)
    local_weights.append(w2)

    # Client 3
    model3 = ParkinsonNN()
    model3.load_state_dict(global_model.state_dict())
    w3 = train_local(model3, X3_train_sc, y3_train)
    local_weights.append(w3)

    # Aggregate
    new_weights = federated_avg(local_weights, data_sizes)
    global_model.load_state_dict(new_weights)

    # ---- Round-wise accuracy on combined TEST set ----
    global_model.eval()

    X_test_all = np.vstack((X1_test_sc, X2_test_sc, X3_test_sc))
    y_test_all = np.hstack((y1_test, y2_test, y3_test))

    with torch.no_grad():
        out     = global_model(torch.tensor(X_test_all, dtype=torch.float32))
        preds_r = torch.argmax(out, dim=1).numpy()

    round_acc = accuracy_score(y_test_all, preds_r)
    round_accuracies.append(round_acc)
    print(f"  Round {r+1} Test Accuracy: {round_acc*100:.2f}%")

    global_model.train()


print("\n" + "="*50)
print("FEDERATED TRAINING COMPLETE")
print("="*50)


# ============================================================
# STEP 6 — Final Evaluation on HELD-OUT TEST DATA
# ============================================================

global_model.eval()

X_test_combined = np.vstack((X1_test_sc, X2_test_sc, X3_test_sc))
y_test_combined = np.hstack((y1_test, y2_test, y3_test))

with torch.no_grad():
    outputs = global_model(torch.tensor(X_test_combined, dtype=torch.float32))
    probs   = torch.softmax(outputs, dim=1).numpy()
    preds   = torch.argmax(outputs, dim=1).numpy()

accuracy    = accuracy_score(y_test_combined, preds)
f1          = f1_score(y_test_combined, preds)
auc         = roc_auc_score(y_test_combined, probs[:, 1])
cm          = confusion_matrix(y_test_combined, preds)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print("\n============================")
print("  FINAL FEDERATED RESULTS   ")
print("============================")
print(f"  Accuracy    : {accuracy*100:.2f}%")
print(f"  F1 Score    : {f1:.4f}")
print(f"  AUC-ROC     : {auc:.4f}")
print(f"  Sensitivity : {sensitivity:.4f}  (PD recall — most important)")
print(f"  Specificity : {specificity:.4f}  (Healthy recall)")
print("\nClassification Report:")
print(classification_report(y_test_combined, preds,
                             target_names=["Healthy", "Parkinson's"]))

print("\nRound-wise Test Accuracy:")
for i, acc in enumerate(round_accuracies):
    print(f"  Round {i+1:2d}: {acc*100:.2f}%")


# ============================================================
# STEP 7 — Save Model + Round Accuracy Log
# ============================================================

torch.save(global_model.state_dict(), "federated_model.pth")
np.save("fl_round_accuracies.npy", np.array(round_accuracies))

print("\nFederated model saved  →  federated_model.pth")
print("Round accuracies saved →  fl_round_accuracies.npy")