"""
cnn_model.py
────────────
Upgraded architecture: CNN + BiLSTM + Attention

Why this is better than the original 2-Conv CNN:

Original:
    Conv → Pool → Conv → Pool → Flatten → FC → FC
    Problem: Flatten destroys temporal structure of spectrogram.
             The model treats time and frequency as flat pixels.
             Hardcoded FC input size (23552) breaks if spectrogram
             shape changes even slightly.

Upgraded — CNN + BiLSTM + Attention:

    Stage 1 — CNN Feature Extractor:
        Extracts local time-frequency patterns from spectrogram.
        Each CNN output frame = compressed representation of a
        short time window.

    Stage 2 — BiLSTM:
        Processes CNN frames as a temporal sequence.
        Forward LSTM  → reads speech left to right
        Backward LSTM → reads right to left
        Together they capture long-range temporal dependencies
        in speech that a flat CNN cannot see.

    Stage 3 — Attention:
        Learns WHICH time steps matter most for diagnosis.
        Assigns a weight to each BiLSTM output frame.
        Weighted sum → single context vector.
        This vector summarizes the most diagnostically
        relevant moments in the voice recording.
        Attention weights are also INTERPRETABLE — you can
        visualize which parts of the speech the model focuses on.

    Stage 4 — Classifier:
        FC layers with Dropout → output logits.

Architecture diagram:
    Input: (batch, 1, 128, T)  ← mel spectrogram
        ↓
    CNN Block 1: Conv2d(1→32)  + BN + ReLU + MaxPool → (batch, 32, 64, T/2)
        ↓
    CNN Block 2: Conv2d(32→64) + BN + ReLU + MaxPool → (batch, 64, 32, T/4)
        ↓
    CNN Block 3: Conv2d(64→128)+ BN + ReLU + MaxPool → (batch, 128, 16, T/8)
        ↓
    Reshape for LSTM: (batch, T/8, 128×16) = (batch, T/8, 2048)
        ↓
    Linear projection: (batch, T/8, 256)   ← reduces LSTM input size
        ↓
    BiLSTM(256→128 each dir): (batch, T/8, 256)
        ↓
    Attention: weighted sum over time → (batch, 256)
        ↓
    FC(256→128) + Dropout(0.4) + ReLU
        ↓
    FC(128→2)   ← output logits
"""

import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# ============================================================
# AUDIO PROCESSING
# ============================================================

def segment_audio(file_path, segment_duration=3, sr=16000):
    """
    Load, normalize, trim silence, split into fixed segments.
    Returns list of numpy arrays.
    """
    y, sr = librosa.load(file_path, sr=sr)
    y     = librosa.util.normalize(y)
    y, _  = librosa.effects.trim(y, top_db=20)

    segment_length = segment_duration * sr
    segments = []

    for start in range(0, len(y), segment_length):
        end = start + segment_length
        if end <= len(y):
            segments.append(y[start:end])

    return segments


def segment_to_mel(segment, sr=16000, n_mels=128, n_fft=512, hop_length=128):
    """
    Convert audio segment → log-mel spectrogram.

    Parameters chosen carefully:
        n_fft=512     → 32ms window at 16kHz — good freq resolution
        hop_length=128 → 8ms step — captures rapid speech changes
        n_mels=128    → matches standard speech processing

    Returns shape: (128, T) where T = ceil(len/hop_length)
    For 3-sec at 16kHz: T ≈ 375 frames
    """
    mel    = librosa.feature.melspectrogram(
        y=segment, sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize to [-1, 1] — stabilizes training
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)

    return mel_db.astype(np.float32)


# ============================================================
# DATASET
# ============================================================

class ParkinsonDataset(Dataset):
    """
    Loads all WAV files from dataset_path/PD/ and dataset_path/HC/
    Segments each file and converts to mel spectrogram.
    Stores (mel, label, person_id) tuples.
    """

    def __init__(self, dataset_path, segment_duration=3):
        self.samples = []
        self.segment_duration = segment_duration

        for label_name in ["PD", "HC"]:
            class_path = os.path.join(dataset_path, label_name)
            label      = 1 if label_name == "PD" else 0

            for person in os.listdir(class_path):
                person_path = os.path.join(class_path, person)
                person_id   = f"{label_name}_{person}"

                # Handle both flat files and person subfolders
                if os.path.isdir(person_path):
                    files = [
                        os.path.join(person_path, f)
                        for f in os.listdir(person_path)
                        if f.endswith(".wav")
                    ]
                elif person_path.endswith(".wav"):
                    files = [person_path]
                else:
                    continue

                for file_path in files:
                    segs = segment_audio(file_path, segment_duration)
                    for seg in segs:
                        mel = segment_to_mel(seg)
                        self.samples.append((mel, label, person_id))

        print(f"Dataset loaded: {len(self.samples)} segments "
              f"from {len(set(s[2] for s in self.samples))} persons")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mel, label, person = self.samples[idx]
        # Add channel dim: (1, n_mels, T)
        mel_tensor   = torch.tensor(mel).unsqueeze(0).float()
        label_tensor = torch.tensor(label).long()
        return mel_tensor, label_tensor, person


# ============================================================
# ATTENTION MODULE
# ============================================================

class BahdanauAttention(nn.Module):
    """
    Additive (Bahdanau) attention over LSTM time steps.

    Given BiLSTM output H of shape (batch, T, hidden_dim):
        score_t = tanh(H_t @ W)  →  scalar per time step
        alpha   = softmax(scores) →  attention weights
        context = Σ alpha_t * H_t →  weighted sum

    The network learns which time frames (speech moments)
    are most informative for PD detection.
    """

    def __init__(self, hidden_dim):
        super(BahdanauAttention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        # lstm_out: (batch, T, hidden_dim)
        scores  = self.attention(lstm_out)          # (batch, T, 1)
        weights = torch.softmax(scores, dim=1)      # (batch, T, 1)
        context = (weights * lstm_out).sum(dim=1)   # (batch, hidden_dim)
        return context, weights.squeeze(-1)         # also return weights for visualization


# ============================================================
# CNN + BiLSTM + ATTENTION MODEL
# ============================================================

class ParkinsonCNN(nn.Module):
    """
    Full architecture: CNN feature extractor → BiLSTM → Attention → FC
    See module docstring for full explanation.
    """

    def __init__(self, n_mels=128, lstm_hidden=128, dropout=0.4):
        super(ParkinsonCNN, self).__init__()

        # ---- CNN Blocks ----
        # Each block: Conv2d + BatchNorm + ReLU + MaxPool
        # Pool only on frequency axis (kernel=(2,1)) in early layers
        # to preserve temporal resolution for LSTM

        self.cnn = nn.Sequential(

            # Block 1: (1, 128, T) → (32, 64, T)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),   # halve frequency only
            nn.Dropout2d(p=0.1),

            # Block 2: (32, 64, T) → (64, 32, T)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout2d(p=0.1),

            # Block 3: (64, 32, T) → (128, 16, T)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout2d(p=0.1),
        )

        # After CNN: shape is (batch, 128, 16, T)
        # Reshape to (batch, T, 128*16) = (batch, T, 2048) for LSTM

        cnn_out_features = 128 * (n_mels // 8)   # = 128 * 16 = 2048

        # ---- Linear projection before LSTM ----
        # Reduces 2048 → 256 to keep LSTM manageable
        self.lstm_proj = nn.Linear(cnn_out_features, 256)

        # ---- BiLSTM ----
        # bidirectional=True → output is 2 * lstm_hidden = 256
        self.bilstm = nn.LSTM(
            input_size   = 256,
            hidden_size  = lstm_hidden,
            num_layers   = 2,
            batch_first  = True,
            bidirectional= True,
            dropout      = 0.3
        )

        lstm_out_dim = lstm_hidden * 2   # 256 (bidirectional)

        # ---- Attention ----
        self.attention = BahdanauAttention(lstm_out_dim)

        # ---- Classifier ----
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 2)
        )

        # ---- Weight initialization ----
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_attention=False):
        """
        x: (batch, 1, n_mels, T)
        return_attention: if True, also returns attention weights
                          (useful for visualization)
        """
        batch_size = x.size(0)

        # Stage 1 — CNN
        x = self.cnn(x)                        # (batch, 128, 16, T)

        # Reshape for LSTM: move time to dim 1
        # (batch, 128, 16, T) → (batch, T, 128*16)
        T = x.size(3)
        x = x.permute(0, 3, 1, 2)             # (batch, T, 128, 16)
        x = x.reshape(batch_size, T, -1)       # (batch, T, 2048)

        # Project down
        x = F.relu(self.lstm_proj(x))          # (batch, T, 256)

        # Stage 2 — BiLSTM
        lstm_out, _ = self.bilstm(x)           # (batch, T, 256)

        # Stage 3 — Attention
        context, attn_weights = self.attention(lstm_out)   # (batch, 256)

        # Stage 4 — Classify
        out = self.classifier(context)         # (batch, 2)

        if return_attention:
            return out, attn_weights
        return out

    def get_architecture_summary(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("\n" + "="*55)
        print("  ParkinsonCNN (CNN + BiLSTM + Attention)")
        print("="*55)
        print("  Input       : (batch, 1, 128, T) mel spectrogram")
        print("  CNN Block 1 : Conv2d(1→32)  + BN + ReLU + Pool(2,1)")
        print("  CNN Block 2 : Conv2d(32→64) + BN + ReLU + Pool(2,1)")
        print("  CNN Block 3 : Conv2d(64→128)+ BN + ReLU + Pool(2,1)")
        print("  Projection  : Linear(2048→256)")
        print("  BiLSTM      : 2 layers, hidden=128, bidirectional")
        print("  Attention   : Bahdanau (additive)")
        print("  Classifier  : FC(256→128→2) + Dropout(0.4)")
        print(f"  Total params    : {total:,}")
        print(f"  Trainable params: {trainable:,}")
        print("="*55 + "\n")


# ============================================================
# SANITY CHECK
# ============================================================

if __name__ == "__main__":
    model = ParkinsonCNN()
    model.get_architecture_summary()

    # Simulate a batch of 4 spectrograms: (batch, 1, 128, 375)
    # 375 ≈ frames for 3-sec audio at hop_length=128
    dummy = torch.randn(4, 1, 128, 375)
    model.eval()

    with torch.no_grad():
        out, attn = model(dummy, return_attention=True)

    print(f"Input shape     : {dummy.shape}")
    print(f"Output shape    : {out.shape}")
    print(f"Attention shape : {attn.shape}")
    print(f"Attention sum   : {attn[0].sum().item():.4f}  (should be ≈ 1.0)")
    print("\nSanity check passed.")