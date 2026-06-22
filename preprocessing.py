import numpy as np
import librosa

# ============================================================
# parselmouth is a Python wrapper around Praat — the gold
# standard tool for voice analysis in clinical research.
# Install with:  pip install praat-parselmouth
# ============================================================
try:
    import parselmouth
    from parselmouth.praat import call
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False
    print(
        "[WARNING] parselmouth not found. Clinical features will be zeros.\n"
        "          Install with: pip install praat-parselmouth\n"
    )


# ============================================================
# FEATURE COUNT REFERENCE
# ============================================================
#
#   Group                       Features    Count
#   ─────────────────────────── ──────────  ─────
#   MFCC (mean + std)           20 × 2      = 40
#   Delta MFCC (mean + std)     20 × 2      = 40
#   Delta-Delta MFCC (mean+std) 20 × 2      = 40
#   Pitch (mean, std)                       =  2
#   Zero Crossing Rate (mean,std)           =  2
#   Spectral Centroid (mean,std)            =  2
#   RMS Energy (mean, std)                  =  2
#   ─────────────────────────────────────────────
#   Existing subtotal                       = 128
#
#   Clinical Features (parselmouth / Praat):
#   Jitter (local, rap, ppq5, ddp)          =  4
#   Shimmer (local, apq3, apq5, apq11, dda) =  5
#   HNR  (Harmonics-to-Noise Ratio)         =  1
#   NHR  (Noise-to-Harmonics Ratio)         =  1
#   F0 stats (mean, std, min, max)          =  4
#   Spectral Flux (mean, std)               =  2
#   Spectral Rolloff (mean, std)            =  2
#   Spectral Bandwidth (mean, std)          =  2
#   ─────────────────────────────────────────────
#   Clinical subtotal                       = 21
#
#   TOTAL FEATURES                          = 149
#
# ============================================================
TOTAL_FEATURES = 149


# ============================================================
# SEGMENT AUDIO
# ============================================================

def segment_audio(file_path, segment_duration=3, sr=16000):
    """
    Load audio, normalize, trim silence, split into fixed segments.
    Returns list of numpy arrays (one per segment).
    """
    y, sr = librosa.load(file_path, sr=sr)
    y     = librosa.util.normalize(y)
    y, _  = librosa.effects.trim(y, top_db=20)

    segment_length = segment_duration * sr
    total_length   = len(y)

    segments = []

    for start in range(0, total_length, segment_length):
        end = start + segment_length
        if end <= total_length:
            segments.append(y[start:end])

    return segments


# ============================================================
# CLINICAL FEATURE EXTRACTION  (parselmouth / Praat)
# ============================================================

def extract_clinical_features(segment, sr=16000):
    """
    Extracts clinically validated Parkinson's biomarkers using Praat.

    These features are used in the original Parkinson's UCI dataset
    (Little et al., 2007) and in subsequent clinical research:

    Jitter    — cycle-to-cycle variation in pitch period
                Elevated in PD due to laryngeal tremor
    Shimmer   — cycle-to-cycle variation in amplitude
                Elevated in PD due to reduced vocal fold control
    HNR/NHR   — ratio of harmonic to noise energy
                Reduced HNR (more noise) is a PD hallmark
    F0 stats  — fundamental frequency statistics
                PD patients show reduced F0 range and stability

    Returns numpy array of 21 values.
    If parselmouth unavailable, returns zeros (graceful fallback).
    """

    if not PARSELMOUTH_AVAILABLE:
        return np.zeros(21)

    try:
        # Convert numpy array → parselmouth Sound object
        snd = parselmouth.Sound(segment, sampling_frequency=sr)

        clinical = []

        # ---- F0 (Fundamental Frequency) ----
        # pitch_floor=75, pitch_ceiling=500 covers adult male/female range
        pitch_obj  = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
        f0_values  = pitch_obj.selected_array['frequency']
        f0_voiced  = f0_values[f0_values > 0]   # remove unvoiced frames

        if len(f0_voiced) > 1:
            clinical.append(np.mean(f0_voiced))    # F0 mean
            clinical.append(np.std(f0_voiced))     # F0 std  — reduced in PD
            clinical.append(np.min(f0_voiced))     # F0 min
            clinical.append(np.max(f0_voiced))     # F0 max — reduced range in PD
        else:
            clinical.extend([0.0, 0.0, 0.0, 0.0])

        # ---- PointProcess (needed for Jitter & Shimmer) ----
        point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)

        # ---- Jitter ----
        # local jitter — most common jitter measure
        jitter_local = call(point_process, "Get jitter (local)",
                            0, 0, 0.0001, 0.02, 1.3)

        # RAP — Relative Average Perturbation
        jitter_rap   = call(point_process, "Get jitter (rap)",
                            0, 0, 0.0001, 0.02, 1.3)

        # PPQ5 — 5-point Period Perturbation Quotient
        jitter_ppq5  = call(point_process, "Get jitter (ppq5)",
                            0, 0, 0.0001, 0.02, 1.3)

        # DDP — Average absolute diff of consecutive differences
        jitter_ddp   = call(point_process, "Get jitter (ddp)",
                            0, 0, 0.0001, 0.02, 1.3)

        clinical.extend([
            jitter_local  if jitter_local  is not None else 0.0,
            jitter_rap    if jitter_rap    is not None else 0.0,
            jitter_ppq5   if jitter_ppq5   is not None else 0.0,
            jitter_ddp    if jitter_ddp    is not None else 0.0
        ])

        # ---- Shimmer ----
        # local shimmer
        shimmer_local = call([snd, point_process], "Get shimmer (local)",
                             0, 0, 0.0001, 0.02, 1.3, 1.6)

        # APQ3 — 3-point Amplitude Perturbation Quotient
        shimmer_apq3  = call([snd, point_process], "Get shimmer (apq3)",
                             0, 0, 0.0001, 0.02, 1.3, 1.6)

        # APQ5
        shimmer_apq5  = call([snd, point_process], "Get shimmer (apq5)",
                             0, 0, 0.0001, 0.02, 1.3, 1.6)

        # APQ11
        shimmer_apq11 = call([snd, point_process], "Get shimmer (apq11)",
                             0, 0, 0.0001, 0.02, 1.3, 1.6)

        # DDA — average absolute diff of consecutive shimmer differences
        shimmer_dda   = call([snd, point_process], "Get shimmer (dda)",
                             0, 0, 0.0001, 0.02, 1.3, 1.6)

        def _safe(v): return v if v is not None else 0.0

        clinical.extend([
            _safe(shimmer_local),
            _safe(shimmer_apq3),
            _safe(shimmer_apq5),
            _safe(shimmer_apq11),
            _safe(shimmer_dda)
        ])

        # ---- HNR (Harmonics-to-Noise Ratio) ----
        # Higher = cleaner voice; reduced in PD
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr         = call(harmonicity, "Get mean", 0, 0)
        clinical.append(_safe(hnr))

        # ---- NHR (Noise-to-Harmonics Ratio) ----
        # Inverse of HNR — elevated in PD
        nhr = 1.0 / hnr if (hnr is not None and hnr > 0) else 0.0
        clinical.append(nhr)

        return np.array(clinical, dtype=np.float32)

    except Exception as e:
        # If Praat fails on a segment (e.g. too short, pure noise),
        # return zeros gracefully — training still continues
        print(f"  [clinical] Praat error on segment: {e}")
        return np.zeros(21)


# ============================================================
# SPECTRAL FEATURES  (librosa — additional 6 features)
# ============================================================

def extract_spectral_features(segment, sr=16000):
    """
    Spectral Flux, Rolloff, Bandwidth — 2 stats each = 6 features.
    These capture frequency distribution changes over time,
    which are altered in PD due to reduced vocal tract control.
    """
    features = []

    # Spectral Flux — measures frame-to-frame spectral change
    # PD patients show irregular spectral flux due to tremor
    onset_env = librosa.onset.onset_strength(y=segment, sr=sr)
    features.append(np.mean(onset_env))
    features.append(np.std(onset_env))

    # Spectral Rolloff — frequency below which 85% of energy lies
    rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr, roll_percent=0.85)
    features.append(np.mean(rolloff))
    features.append(np.std(rolloff))

    # Spectral Bandwidth — spread of frequencies around centroid
    bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
    features.append(np.mean(bandwidth))
    features.append(np.std(bandwidth))

    return np.array(features, dtype=np.float32)


# ============================================================
# MAIN FEATURE EXTRACTION — combines all feature groups
# ============================================================

def extract_features(segment, sr=16000):
    """
    Full feature extraction pipeline.

    Returns numpy array of 149 features:
        [0:128]   — existing MFCC + acoustic features
        [128:149] — clinical features (Praat) + spectral

    The feature count is defined in TOTAL_FEATURES = 149.
    All models (SVM, RF, FL) must use this same function.
    """
    features = []

    # =========================================================
    # GROUP 1 — MFCC Features (40 features)
    # =========================================================
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=20)
    features.extend(np.mean(mfcc, axis=1))   # 20
    features.extend(np.std(mfcc, axis=1))    # 20

    # =========================================================
    # GROUP 2 — Delta MFCC (40 features)
    # =========================================================
    delta_mfcc = librosa.feature.delta(mfcc)
    features.extend(np.mean(delta_mfcc, axis=1))
    features.extend(np.std(delta_mfcc, axis=1))

    # =========================================================
    # GROUP 3 — Delta-Delta MFCC (40 features)
    # =========================================================
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    features.extend(np.mean(delta2_mfcc, axis=1))
    features.extend(np.std(delta2_mfcc, axis=1))

    # =========================================================
    # GROUP 4 — Pitch / F0 via librosa (2 features)
    # =========================================================
    pitches, magnitudes = librosa.piptrack(y=segment, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]

    if len(pitch_values) > 0:
        features.append(np.mean(pitch_values))
        features.append(np.std(pitch_values))
    else:
        features.append(0.0)
        features.append(0.0)

    # =========================================================
    # GROUP 5 — Zero Crossing Rate (2 features)
    # =========================================================
    zcr = librosa.feature.zero_crossing_rate(segment)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # =========================================================
    # GROUP 6 — Spectral Centroid (2 features)
    # =========================================================
    centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
    features.append(np.mean(centroid))
    features.append(np.std(centroid))

    # =========================================================
    # GROUP 7 — RMS Energy (2 features)
    # =========================================================
    rms = librosa.feature.rms(y=segment)
    features.append(np.mean(rms))
    features.append(np.std(rms))

    # ── subtotal so far: 128 ──────────────────────────────────

    # =========================================================
    # GROUP 8 — Clinical Features via Praat (15 features)
    #   F0 stats (4) + Jitter (4) + Shimmer (5) + HNR + NHR
    # =========================================================
    clinical = extract_clinical_features(segment, sr=sr)
    features.extend(clinical)

    # =========================================================
    # GROUP 9 — Additional Spectral Features (6 features)
    #   Spectral Flux, Rolloff, Bandwidth  (mean + std each)
    # =========================================================
    spectral = extract_spectral_features(segment, sr=sr)
    features.extend(spectral)

    # ── total: 149 ───────────────────────────────────────────

    result = np.array(features, dtype=np.float32)

    # Safety check — catches bugs during development
    assert result.shape[0] == TOTAL_FEATURES, (
        f"Feature count mismatch: expected {TOTAL_FEATURES}, "
        f"got {result.shape[0]}"
    )

    return result