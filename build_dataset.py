"""
build_dataset.py  —  FAST VERSION
──────────────────────────────────
Speed fixes vs previous version:

  1. Praat (clinical features) runs on ORIGINAL segments only
     — Jitter/Shimmer on noise-augmented audio is unreliable anyway
     — Augmented segments reuse the original's clinical features
     — This alone cuts runtime by ~75%

  2. Augmentation only modifies the audio BEFORE MFCC extraction
     — We do NOT re-run Praat per augmented segment
     — Augmented segment gets MFCC/spectral features recalculated
       (cheap, librosa only) + reuses original clinical features

  3. Progress printed every 10 persons so you can see it moving

  4. try/except per segment so one bad file doesn't stop the run

Expected runtime:
  Before : 30+ minutes (Praat on every augmented segment)
  After  : 4–8 minutes  (Praat only on originals)
"""

import os
import numpy as np
import librosa
from preprocessing import segment_audio, extract_features, TOTAL_FEATURES


# ============================================================
# AUGMENTATION  (audio-level, no Praat)
# ============================================================

def aug_noise(seg):
    return (seg + 0.005 * np.random.randn(len(seg))).astype(np.float32)

def aug_stretch(seg, rate=0.9):
    out = librosa.effects.time_stretch(seg, rate=rate)
    n   = len(seg)
    return out[:n] if len(out) >= n else np.pad(out, (0, n - len(out)))

def aug_pitch(seg, sr=16000, steps=2):
    return librosa.effects.pitch_shift(seg, sr=sr, n_steps=steps).astype(np.float32)


# ============================================================
# ACOUSTIC-ONLY FEATURES  (librosa, no Praat — fast)
# ============================================================

def extract_acoustic_only(segment, sr=16000):
    """
    Extracts only the 128 acoustic features — no Praat.
    Used for augmented segments.
    """
    features = []

    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=20)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    delta_mfcc = librosa.feature.delta(mfcc)
    features.extend(np.mean(delta_mfcc, axis=1))
    features.extend(np.std(delta_mfcc, axis=1))

    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    features.extend(np.mean(delta2_mfcc, axis=1))
    features.extend(np.std(delta2_mfcc, axis=1))

    pitches, magnitudes = librosa.piptrack(y=segment, sr=sr)
    pv = pitches[magnitudes > np.median(magnitudes)]
    features.append(np.mean(pv) if len(pv) > 0 else 0.0)
    features.append(np.std(pv)  if len(pv) > 0 else 0.0)

    zcr = librosa.feature.zero_crossing_rate(segment)
    features.append(np.mean(zcr)); features.append(np.std(zcr))

    centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
    features.append(np.mean(centroid)); features.append(np.std(centroid))

    rms = librosa.feature.rms(y=segment)
    features.append(np.mean(rms)); features.append(np.std(rms))

    return np.array(features, dtype=np.float32)   # (128,)


# ============================================================
# ASSEMBLE AUGMENTED FEATURE VECTOR
# New acoustic (128) + original clinical (15) + new spectral (6) = 149
# ============================================================

def make_aug_features(aug_seg, orig_full_features, sr=16000):
    """
    Builds a 149-feature vector for an augmented segment.

    Layout (matches preprocessing.py extract_features):
      [0:128]   acoustic  — recalculated fresh (captures augmentation effect)
      [128:143] clinical  — REUSED from original (Praat not re-run)
      [143:149] spectral  — recalculated fresh

    Key insight: augmentation affects MFCC/spectral features,
    which the model uses for temporal pattern learning.
    Clinical features (Jitter/Shimmer) on augmented audio are
    unreliable artifacts — better to reuse the original values.
    """
    # Fresh acoustic features for the augmented audio
    acoustic = extract_acoustic_only(aug_seg, sr=sr)       # (128,)

    # Reuse clinical features from original segment
    clinical = orig_full_features[128:143].copy()          # (15,)

    # Fresh spectral features
    onset_env = librosa.onset.onset_strength(y=aug_seg, sr=sr)
    rolloff   = librosa.feature.spectral_rolloff(y=aug_seg, sr=sr, roll_percent=0.85)
    bandwidth = librosa.feature.spectral_bandwidth(y=aug_seg, sr=sr)

    spectral = np.array([
        np.mean(onset_env), np.std(onset_env),
        np.mean(rolloff),   np.std(rolloff),
        np.mean(bandwidth), np.std(bandwidth)
    ], dtype=np.float32)                                    # (6,)

    full = np.concatenate([acoustic, clinical, spectral])  # (149,)

    assert full.shape[0] == TOTAL_FEATURES, \
        f"Aug feature mismatch: {full.shape[0]} vs {TOTAL_FEATURES}"

    return full


# ============================================================
# DATASET BUILDER  (one class, one dataset at a time)
# ============================================================

def build_dataset(dataset_path, label_name, seg_duration=3,
                  use_augmentation=True, sr=16000):
    """
    Builds features for one label (PD or HC) from one dataset folder.
    Returns (X_list, y_list, person_id_list).
    """
    X_out   = []
    y_out   = []
    pid_out = []

    label      = 1 if label_name == "PD" else 0
    class_path = os.path.join(dataset_path, label_name)

    if not os.path.exists(class_path):
        print(f"  [skip] Path not found: {class_path}")
        return X_out, y_out, pid_out

    persons   = sorted(os.listdir(class_path))
    n_persons = len(persons)

    for p_idx, item in enumerate(persons):

        # Progress update every 5 persons
        if p_idx % 5 == 0:
            print(f"    [{label_name}] {p_idx+1}/{n_persons} ...", flush=True)

        item_path = os.path.join(class_path, item)
        person_id = f"{label_name}_{os.path.splitext(item)[0]}"

        # Collect WAV files (handles flat files and subfolders)
        wav_files = []
        if item.endswith(".wav"):
            wav_files = [item_path]
        elif os.path.isdir(item_path):
            person_id = f"{label_name}_{item}"
            wav_files = [
                os.path.join(item_path, f)
                for f in os.listdir(item_path)
                if f.endswith(".wav")
            ]

        for fpath in wav_files:
            try:
                segments = segment_audio(
                    fpath, segment_duration=seg_duration, sr=sr
                )
            except Exception as e:
                print(f"      [skip file] {os.path.basename(fpath)}: {e}")
                continue

            for seg in segments:

                # ---- Original: full 149-feature extraction (Praat runs here) ----
                try:
                    orig_feat = extract_features(seg, sr=sr)
                    X_out.append(orig_feat)
                    y_out.append(label)
                    pid_out.append(person_id)
                except Exception as e:
                    print(f"      [skip seg original] {e}")
                    continue

                if not use_augmentation:
                    continue

                # ---- Augmented: acoustic recalculated, clinical REUSED ----
                # Praat does NOT run again — this is what makes it fast
                aug_list = [
                    ("noise",   aug_noise(seg)),
                    ("stretch", aug_stretch(seg, rate=0.9)),
                    ("pitch",   aug_pitch(seg, sr=sr, steps=2)),
                ]

                for aug_name, aug_seg in aug_list:
                    try:
                        aug_feat = make_aug_features(aug_seg, orig_feat, sr=sr)
                        X_out.append(aug_feat)
                        y_out.append(label)
                        pid_out.append(person_id)   # same person_id — no leakage
                    except Exception as e:
                        print(f"      [skip aug {aug_name}] {e}")
                        continue

    return X_out, y_out, pid_out


# ============================================================
# BUILD ALL THREE DATASETS
# ============================================================

USE_AUGMENTATION = True

# (path, segment_duration_seconds)
DATASETS = [
    ("D:/Finall/datasett", 3),   # Italian — multiple speech types
    ("D:/Finall/dataset2", 1),   # 41+41   — short vowel sounds
    ("D:/Finall/dataset3", 3),   # Large   — general recordings
]

all_X   = []
all_y   = []
all_pid = []

for ds_path, seg_dur in DATASETS:
    ds_name = os.path.basename(ds_path)

    print(f"\n{'='*50}")
    print(f"  Dataset : {ds_name}  |  seg={seg_dur}s  |  aug={USE_AUGMENTATION}")
    print(f"{'='*50}")

    for lbl in ["PD", "HC"]:
        X_p, y_p, pid_p = build_dataset(
            ds_path, lbl,
            seg_duration     = seg_dur,
            use_augmentation = USE_AUGMENTATION
        )
        all_X.extend(X_p)
        all_y.extend(y_p)
        all_pid.extend(pid_p)

    print(f"  Running total: {len(all_X)} segments so far")


# ============================================================
# STACK & EXPORT
# ============================================================

X          = np.array(all_X,   dtype=np.float32)
y          = np.array(all_y,   dtype=np.int32)
person_ids = np.array(all_pid, dtype=str)

pd_count = int(np.sum(y == 1))
hc_count = int(np.sum(y == 0))
nan_vals = int(np.isnan(X).sum())

print(f"\n{'='*50}")
print(f"  DATASET READY")
print(f"{'='*50}")
print(f"  Shape          : {X.shape}")
print(f"  Unique persons : {len(np.unique(person_ids))}")
print(f"  PD samples     : {pd_count}  ({pd_count/len(y)*100:.1f}%)")
print(f"  HC samples     : {hc_count}  ({hc_count/len(y)*100:.1f}%)")
print(f"  NaN values     : {nan_vals}  "
      f"{'← will be handled by imputer in train_ml.py' if nan_vals > 0 else '← clean!'}")
print(f"  Features       : {X.shape[1]}  (expected {TOTAL_FEATURES})")

assert X.shape[1] == TOTAL_FEATURES, \
    f"Feature count wrong: {X.shape[1]} vs {TOTAL_FEATURES}"

print(f"\n  ✓ Verified. Ready for train_ml.py")