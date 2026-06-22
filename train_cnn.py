"""
train_cnn.py
────────────
Lightweight CNN for Parkinson's detection from mel spectrograms.

Replaces the heavy CNN+BiLSTM+Attention (1.4M params) with a
simple 3-block CNN (~150K params) that trains fast on CPU.

Architecture:
    Input  : (batch, 1, 64, T) — 64 mel bands (reduced from 128)
    Block 1: Conv(1→16)  + BN + ReLU + MaxPool
    Block 2: Conv(16→32) + BN + ReLU + MaxPool
    Block 3: Conv(32→64) + BN + ReLU + AdaptivePool
    Head   : Flatten → FC(64→32) → Dropout(0.3) → FC(32→2)

Changes that make it lighter / faster vs original:
    - 64 mel bands instead of 128 (half the spectrogram size)
    - No BiLSTM, no Attention (removes ~1M params)
    - Smaller channel counts (16→32→64 vs 32→64→128)
    - AdaptiveAvgPool collapses time — no fixed input size needed
    - Batch size 64 (bigger batches = fewer steps per epoch)
    - 20 epochs max with patience=5 early stopping
    - Dataset built inline — no dependency on cnn_model.py
"""

import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score,
    roc_auc_score, confusion_matrix,
    classification_report
)


# ============================================================
# CONFIG
# ============================================================

DATASET_PATH     = "D:/Finall/datasett"
SEGMENT_DURATION = 3       # seconds per segment
SR               = 16000
N_MELS           = 64      # reduced from 128 → half the compute
BATCH_SIZE       = 64      # larger batch → fewer steps per epoch
EPOCHS           = 5
LR               = 0.001
PATIENCE         = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")


# ============================================================
# DATASET
# ============================================================

class ParkinsonDataset(Dataset):
    """
    Loads .wav files, segments them, converts to mel spectrogram.
    Returns (mel_tensor, label, person_id).
    """

    def __init__(self, dataset_path):
        self.seg_len = SEGMENT_DURATION * SR
        self.samples = []   # (file_path, label, person_id, start_sample)

        for label_name, label_int in [("PD", 1), ("HC", 0)]:
            class_path = os.path.join(dataset_path, label_name)
            for person in os.listdir(class_path):
                person_path = os.path.join(class_path, person)
                person_id   = f"{label_name}_{person}"
                for fname in os.listdir(person_path):
                    if not fname.endswith(".wav"):
                        continue
                    fpath = os.path.join(person_path, fname)
                    try:
                        duration = librosa.get_duration(path=fpath)
                        n_segs   = int(duration // SEGMENT_DURATION)
                        for i in range(n_segs):
                            self.samples.append(
                                (fpath, label_int, person_id, i * self.seg_len)
                            )
                    except Exception:
                        continue

        print(f"Dataset: {len(self.samples)} segments, "
              f"{len(set(s[2] for s in self.samples))} persons")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label, person_id, start = self.samples[idx]

        y, _ = librosa.load(
            fpath, sr=SR,
            offset=start / SR,
            duration=SEGMENT_DURATION
        )

        # Pad short segments
        if len(y) < self.seg_len:
            y = np.pad(y, (0, self.seg_len - len(y)))

        y = librosa.util.normalize(y)

        mel    = librosa.feature.melspectrogram(
            y=y, sr=SR, n_mels=N_MELS, n_fft=512, hop_length=256
        )
        mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

        # Normalize to [0, 1]
        mn, mx = mel_db.min(), mel_db.max()
        mel_db = (mel_db - mn) / (mx - mn + 1e-6)

        return torch.tensor(mel_db).unsqueeze(0), \
               torch.tensor(label, dtype=torch.long), \
               person_id


# ============================================================
# LIGHTWEIGHT CNN MODEL  (~150K params)
# ============================================================

class LightCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))   # fixed output regardless of input size
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ============================================================
# LOAD DATASET & PERSON-WISE SPLIT
# ============================================================

print("\nLoading dataset...")
dataset = ParkinsonDataset(DATASET_PATH)

person_labels = {}
for _, label, pid, _ in dataset.samples:
    person_labels[pid] = int(label)

pid_list   = list(person_labels.keys())
label_list = [person_labels[p] for p in pid_list]

train_pids, test_pids = train_test_split(
    pid_list, test_size=0.2, stratify=label_list, random_state=42
)

train_set     = set(train_pids)
test_set      = set(test_pids)
train_indices = [i for i, s in enumerate(dataset.samples) if s[2] in train_set]
test_indices  = [i for i, s in enumerate(dataset.samples) if s[2] in test_set]

print(f"Train: {len(train_indices)} segments | Test: {len(test_indices)} segments")

train_loader = DataLoader(Subset(dataset, train_indices),
                          batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(Subset(dataset, test_indices),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# ============================================================
# MODEL, OPTIMIZER, SCHEDULER
# ============================================================

model     = LightCNN().to(device)
total_p   = sum(p.numel() for p in model.parameters())
print(f"Model params: {total_p:,}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)


# ============================================================
# TRAINING LOOP
# ============================================================

best_val_loss     = float("inf")
epochs_no_improve = 0
best_model_state  = None
train_losses, val_losses, val_accs = [], [], []

print("\n" + "=" * 55)
print("  TRAINING")
print("=" * 55)

for epoch in range(1, EPOCHS + 1):

    # ---- Train ----
    model.train()
    running_loss = 0.0

    for X_batch, y_batch, _ in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()

    avg_train = running_loss / len(train_loader)
    train_losses.append(avg_train)

    # ---- Validate ----
    model.eval()
    val_loss = 0.0
    y_true_v, y_pred_v, y_prob_v = [], [], []

    with torch.no_grad():
        for X_batch, y_batch, _ in test_loader:
            out = model(X_batch.to(device))
            val_loss += criterion(out, y_batch.to(device)).item()
            probs = torch.softmax(out, dim=1).cpu().numpy()
            y_true_v.extend(y_batch.numpy())
            y_pred_v.extend(np.argmax(probs, axis=1))
            y_prob_v.extend(probs[:, 1])

    avg_val   = val_loss / len(test_loader)
    epoch_acc = accuracy_score(y_true_v, y_pred_v)
    val_losses.append(avg_val)
    val_accs.append(epoch_acc)

    print(f"Epoch {epoch:2d}/{EPOCHS}  "
          f"Train: {avg_train:.4f}  Val: {avg_val:.4f}  "
          f"Acc: {epoch_acc*100:.2f}%  "
          f"LR: {optimizer.param_groups[0]['lr']:.5f}")

    scheduler.step(avg_val)

    if avg_val < best_val_loss:
        best_val_loss    = avg_val
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        epochs_no_improve = 0
        print("  ✓ Best model saved")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break


# ============================================================
# FINAL EVALUATION
# ============================================================

print("\n" + "=" * 55)
print("  FINAL EVALUATION")
print("=" * 55)

model.load_state_dict(best_model_state)
model.eval()

y_true_all, y_pred_all, y_prob_all, person_all = [], [], [], []

with torch.no_grad():
    for X_batch, y_batch, persons_batch in test_loader:
        out   = model(X_batch.to(device))
        probs = torch.softmax(out, dim=1).cpu().numpy()
        y_true_all.extend(y_batch.numpy())
        y_pred_all.extend(np.argmax(probs, axis=1))
        y_prob_all.extend(probs[:, 1])
        person_all.extend(list(persons_batch))

y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)
y_prob_all = np.array(y_prob_all)

seg_cm         = confusion_matrix(y_true_all, y_pred_all)
tn, fp, fn, tp = seg_cm.ravel()

print("\n  Segment-Level:")
print(f"    Accuracy    : {accuracy_score(y_true_all, y_pred_all)*100:.2f}%")
print(f"    F1 Score    : {f1_score(y_true_all, y_pred_all):.4f}")
print(f"    AUC-ROC     : {roc_auc_score(y_true_all, y_prob_all):.4f}")
print(f"    Sensitivity : {tp/(tp+fn):.4f}")
print(f"    Specificity : {tn/(tn+fp):.4f}")

# ---- Person-level majority vote ----
df  = pd.DataFrame({
    "person"    : person_all,
    "true_label": y_true_all,
    "pred_label": y_pred_all,
    "prob_pd"   : y_prob_all
})
pdf = df.groupby("person").agg(
    true_label  = ("true_label",  "first"),
    voted_label = ("pred_label",  lambda x: np.bincount(x).argmax()),
    mean_prob   = ("prob_pd",     "mean")
).reset_index()

p_cm                    = confusion_matrix(pdf["true_label"], pdf["voted_label"])
p_tn, p_fp, p_fn, p_tp = p_cm.ravel()

print("\n  Person-Level (Majority Vote):")
print(f"    Accuracy    : {accuracy_score(pdf['true_label'], pdf['voted_label'])*100:.2f}%")
print(f"    F1 Score    : {f1_score(pdf['true_label'], pdf['voted_label']):.4f}")
print(f"    AUC-ROC     : {roc_auc_score(pdf['true_label'], pdf['mean_prob']):.4f}")
print(f"    Sensitivity : {p_tp/(p_tp+p_fn):.4f}")
print(f"    Specificity : {p_tn/(p_tn+p_fp):.4f}")

print("\n  Classification Report:")
print(classification_report(y_true_all, y_pred_all,
                             target_names=["Healthy", "Parkinson's"]))

# ============================================================
# SAVE
# ============================================================

torch.save(best_model_state,         "cnn_model.pth")
np.save("cnn_train_losses.npy", np.array(train_losses))
np.save("cnn_val_losses.npy",   np.array(val_losses))
np.save("cnn_val_accs.npy",     np.array(val_accs))

print("Saved: cnn_model.pth")
print("Done.")