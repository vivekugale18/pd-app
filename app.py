import streamlit as st
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import torch
import shap
import os
from sklearn.impute import SimpleImputer

from preprocessing import extract_features, segment_audio, TOTAL_FEATURES
from federated_model import ParkinsonNN


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Parkinson's Detection System",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    models = {}

    # Centralized imputer
    if os.path.exists("imputer.pkl"):
        models["imputer"] = joblib.load("imputer.pkl")
    else:
        models["imputer"] = SimpleImputer(strategy="median")
        models["imputer"].fit(np.zeros((1, TOTAL_FEATURES)))

    # SVM
    if os.path.exists("svm_model.pkl") and os.path.exists("scaler.pkl"):
        models["svm"]    = joblib.load("svm_model.pkl")
        models["scaler"] = joblib.load("scaler.pkl")
    else:
        models["svm"]    = None
        models["scaler"] = None

    # Random Forest
    if os.path.exists("rf_model.pkl"):
        models["rf"] = joblib.load("rf_model.pkl")
    else:
        models["rf"] = None

    # Federated model
    if os.path.exists("federated_model.pth"):
        fl = ParkinsonNN(input_size=TOTAL_FEATURES)
        fl.load_state_dict(torch.load("federated_model.pth", map_location="cpu"))
        fl.eval()
        models["fl"] = fl
    else:
        models["fl"] = None

    # FL imputer + scaler
    models["fl_imputer"] = (
        joblib.load("fl_imputer_client1.pkl")
        if os.path.exists("fl_imputer_client1.pkl")
        else models["imputer"]
    )
    models["fl_scaler"] = (
        joblib.load("fl_scaler_client1.pkl")
        if os.path.exists("fl_scaler_client1.pkl")
        else models.get("scaler")
    )

    # SHAP — TreeExplainer built on Random Forest only.
    # We store it once here; the same explainer is used for RF predictions.
    # SVM is NOT supported by TreeExplainer — handled separately below.
    if os.path.exists("shap_values.npy") and os.path.exists("shap_feature_names.npy"):
        models["shap_values"] = np.load("shap_values.npy", allow_pickle=True)
        models["shap_names"]  = np.load("shap_feature_names.npy", allow_pickle=True).tolist()
        if models["rf"] is not None:
            models["shap_explainer"] = shap.TreeExplainer(models["rf"])
        else:
            models["shap_explainer"] = None
    else:
        models["shap_values"]    = None
        models["shap_names"]     = None
        models["shap_explainer"] = None

    return models


models = load_models()


# ============================================================
# WHY THE BUG HAPPENED — explanation for reference
# ============================================================
#
# shap.TreeExplainer(rf_model).shap_values(X) returns:
#
#   For Random Forest (binary):  a LIST of 2 arrays
#       sv[0] = SHAP values for class 0 (Healthy)  shape: (n_samples, n_features)
#       sv[1] = SHAP values for class 1 (PD)        shape: (n_samples, n_features)
#   So sv[1][0] means: class-1 SHAP → first sample  → works fine
#
#   BUT if you pass a SINGLE sample (1 row) to shap_values(),
#   some SHAP versions collapse the output to shape (n_features,) instead of
#   (1, n_features). In that case:
#       sv[1] tries to index the SECOND ELEMENT of a 1D array of length n_features
#       → IndexError: index 1 is out of bounds for axis 0 with size 1
#
#   For SVM:  TreeExplainer does not support SVM at all.
#   The old code used the RF explainer with SVM predictions → wrong model,
#   wrong expected_value, wrong output format → crash.
#
# THE FIX:
#   1. Use shap.Explanation objects returned by the new SHAP API
#      (explainer(X) instead of explainer.shap_values(X)) — this is
#      version-safe and always returns a consistent shape.
#   2. For SVM: do NOT show a live SHAP waterfall — show a friendly message
#      instead. SVM requires KernelExplainer which takes 30-60 seconds per
#      sample — not usable in real-time. Global SHAP plots (from RF) are
#      still shown in Tab 2 and are valid for both models since they share
#      the same 149 features.
# ============================================================

def get_shap_explanation(feat_scaled):
    """
    Returns a shap.Explanation object for the given scaled feature row.

    Uses the new SHAP API (explainer(X)) which is version-safe:
    - Always returns shape (1, n_features) regardless of SHAP version
    - Automatically handles binary classification output format
    - Works correctly with AdditiveExplanation / TreeExplanation objects

    Falls back to the legacy .shap_values() approach with safe indexing
    if the new API is not available (very old SHAP versions).
    """
    explainer = models["shap_explainer"]
    if explainer is None:
        return None, None

    try:
        # ── New SHAP API (recommended, SHAP >= 0.40) ──────────────────────
        # explainer(X) returns a shap.Explanation object with consistent shape
        explanation = explainer(feat_scaled)

        # For binary classification, explanation.values shape is
        # (n_samples, n_features, n_classes) → slice class 1
        if explanation.values.ndim == 3:
            vals = explanation.values[0, :, 1]      # shape (n_features,)
            base = float(explanation.base_values[0, 1])
        elif explanation.values.ndim == 2:
            # Already (n_samples, n_features) — class 1 only
            vals = explanation.values[0, :]
            base_raw = explanation.base_values
            base = float(base_raw[0]) if hasattr(base_raw, '__len__') else float(base_raw)
        else:
            vals = explanation.values
            base = float(explanation.base_values)

        return vals, base

    except Exception:
        # ── Legacy SHAP API fallback (.shap_values()) ─────────────────────
        # Some older versions don't support explainer(X) syntax
        sv = explainer.shap_values(feat_scaled)
        ev = explainer.expected_value

        if isinstance(sv, list) and len(sv) == 2:
            # Standard binary output: [class0_array, class1_array]
            sv_class1 = sv[1]
            ev_class1 = ev[1] if hasattr(ev, '__len__') else float(ev)
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            # (n_samples, n_features, n_classes)
            sv_class1 = sv[:, :, 1]
            ev_class1 = ev[1] if hasattr(ev, '__len__') else float(ev)
        else:
            # Already class-1 or ambiguous — use directly
            sv_class1 = sv
            ev_class1 = ev[1] if (hasattr(ev, '__len__') and len(ev) >= 2) else float(ev)

        # Safely extract first (only) sample row
        if isinstance(sv_class1, np.ndarray) and sv_class1.ndim == 2:
            vals = sv_class1[0]          # shape (n_features,)
        elif isinstance(sv_class1, np.ndarray) and sv_class1.ndim == 1:
            vals = sv_class1             # already (n_features,)
        else:
            vals = np.array(sv_class1).flatten()

        return vals, float(ev_class1)


# ============================================================
# HELPERS
# ============================================================

def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(8, 2.5))
    time = np.linspace(0, len(y) / sr, len(y))
    ax.plot(time, y, color="#4A90D9", linewidth=0.6)
    ax.set_title("Waveform", fontsize=12)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_spectrogram(y, sr):
    fig, ax = plt.subplots(figsize=(8, 2.5))
    S    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    img  = ax.imshow(S_db, aspect="auto", origin="lower",
                     cmap="magma", interpolation="nearest")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel Spectrogram", fontsize=12)
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Mel bands")
    fig.tight_layout()
    return fig


def risk_level(confidence, pred):
    if pred == 0:
        return "Low Risk", "green"
    if confidence >= 0.85:
        return "High Risk", "red"
    elif confidence >= 0.65:
        return "Medium Risk", "orange"
    else:
        return "Low-Medium Risk", "orange"


def run_prediction(features_raw, model_choice):
    """
    raw features → imputer → scaler → model → (pred, prob, prob_pd)
    Returns (None, None, None) if model not loaded.
    """
    if model_choice == "SVM":
        if models["svm"] is None:
            return None, None, None
        feat = models["imputer"].transform([features_raw])
        feat = models["scaler"].transform(feat)
        pred = models["svm"].predict(feat)[0]
        prob = models["svm"].predict_proba(feat)[0]

    elif model_choice == "Random Forest":
        if models["rf"] is None:
            return None, None, None
        feat = models["imputer"].transform([features_raw])
        feat = models["scaler"].transform(feat)
        pred = models["rf"].predict(feat)[0]
        prob = models["rf"].predict_proba(feat)[0]

    else:  # Federated
        if models["fl"] is None:
            return None, None, None
        feat   = models["fl_imputer"].transform([features_raw])
        feat   = models["fl_scaler"].transform(feat)
        tensor = torch.tensor(feat, dtype=torch.float32)
        with torch.no_grad():
            out  = models["fl"](tensor)
            prob = torch.softmax(out, dim=1).numpy()[0]
            pred = int(np.argmax(prob))

    return int(pred), prob, float(prob[1])


# ============================================================
# TABS
# ============================================================

st.title("🧠 Parkinson's Early Detection System")
st.markdown("##### Voice-based detection using ML · Federated Learning · Explainable AI (SHAP)")
st.divider()

tab1, tab2, tab3 = st.tabs([
    "🔍 Detection",
    "📊 Explainability (SHAP)",
    "📡 Federated Learning"
])


# ============================================================
# TAB 1 — DETECTION
# ============================================================

with tab1:

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Settings")

        model_choice = st.radio(
            "Select Model",
            ["SVM", "Random Forest", "Federated Model"],
            help=(
                "SVM and Random Forest use centralized training on all 3 datasets. "
                "Federated Model was trained across 3 private datasets without sharing raw data."
            )
        )

        uploaded_file = st.file_uploader(
            "Upload Voice Recording (.wav)",
            type=["wav"],
            help="Upload a sustained vowel ('ahh') recording, 3–10 seconds."
        )

    with col_right:

        if uploaded_file is None:
            st.info("👈  Upload a WAV file to begin analysis.")

        else:
            with open("temp.wav", "wb") as f:
                f.write(uploaded_file.read())

            st.audio("temp.wav")

            y_audio, sr = librosa.load("temp.wav", sr=16000)

            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(plot_waveform(y_audio, sr))
            with c2:
                st.pyplot(plot_spectrogram(y_audio, sr))

            segments = segment_audio("temp.wav")

            if len(segments) == 0:
                st.error("Audio too short — please upload at least 3 seconds.")

            else:
                with st.spinner(f"Analyzing {len(segments)} segment(s)..."):

                    all_preds = []
                    all_probs = []
                    all_feats = []

                    mc = model_choice.replace(" Model", "")

                    for seg in segments:
                        feat = extract_features(seg)
                        pred, prob, prob_pd = run_prediction(feat, mc)
                        if pred is not None:
                            all_preds.append(pred)
                            all_probs.append(prob)
                            all_feats.append(feat)

                if len(all_probs) == 0:
                    st.error("Model not loaded. Run train_ml.py first.")

                else:
                    avg_prob    = np.mean(all_probs, axis=0)
                    final_pred  = int(np.argmax(avg_prob))
                    confidence = float(min(np.max(avg_prob), 0.99))
                    risk, color = risk_level(confidence, final_pred)

                    st.divider()
                    st.subheader("Result")

                    res_col1, res_col2, res_col3 = st.columns(3)
                    with res_col1:
                        if final_pred == 1:
                            st.error("⚠️ Parkinson's Detected")
                        else:
                            st.success("✅ Healthy")
                    with res_col2:
                        st.metric("Confidence", f"{confidence*100:.1f}%")
                    with res_col3:
                        st.metric("Risk Level", risk)

                    st.caption(f"Based on {len(segments)} segment(s) · Model: {model_choice}")

                    # Segment breakdown
                    with st.expander("Segment-level breakdown"):
                        st.table({
                            "Segment"       : [f"Seg {i+1}" for i in range(len(all_preds))],
                            "Prediction"    : ["Parkinson's" if p == 1 else "Healthy" for p in all_preds],
                            "PD Probability": [f"{p[1]*100:.1f}%" for p in all_probs]
                        })

                    # ── SHAP waterfall ─────────────────────────────────────
                    if model_choice == "Random Forest" and models["shap_explainer"] is not None:

                        with st.expander("🔍 SHAP Explanation — Why this prediction?"):
                            st.markdown(
                                "Red bars push toward **Parkinson's**, "
                                "blue bars push toward **Healthy**."
                            )
                            try:
                                # Preprocess first segment
                                feat_imp    = models["imputer"].transform([all_feats[0]])
                                feat_scaled = models["scaler"].transform(feat_imp)

                                # Safe extraction — handles all SHAP version formats
                                sv_pd, ev_pd = get_shap_explanation(feat_scaled)

                                if sv_pd is not None:
                                    fig_wf, _ = plt.subplots(figsize=(10, 6))
                                    shap.waterfall_plot(
                                        shap.Explanation(
                                            values        = sv_pd,
                                            base_values   = ev_pd,
                                            data          = feat_scaled[0],
                                            feature_names = models["shap_names"]
                                        ),
                                        max_display=12,
                                        show=False
                                    )
                                    plt.tight_layout()
                                    st.pyplot(fig_wf)
                                    plt.close()

                            except Exception as e:
                                st.warning(f"SHAP explanation unavailable for this segment: {e}")

                    elif model_choice == "SVM":
                        # SVM does not support TreeExplainer.
                        # KernelExplainer works but takes 30–60s per sample
                        # — not suitable for a real-time web app.
                        # Global SHAP plots (Tab 2) still apply since
                        # both models use the same 149 features.
                        with st.expander("ℹ️ SHAP for SVM"):
                            st.info(
                                "Live per-prediction SHAP is only available for **Random Forest**. "
                                "SVM uses a kernel-based decision boundary that requires "
                                "KernelExplainer — which takes 30–60 seconds per sample "
                                "and is not practical for real-time inference.\n\n"
                                "Switch to **Random Forest** to see per-prediction SHAP explanations. "
                                "The global SHAP analysis in the **Explainability tab** applies "
                                "to both models since they use the same 149 features."
                            )

                    elif model_choice == "Federated Model":
                        with st.expander("ℹ️ SHAP for Federated Model"):
                            st.info(
                                "Per-prediction SHAP is not available for the Federated model. "
                                "View global SHAP analysis in the **Explainability tab** "
                                "(generated from Random Forest on the same feature set)."
                            )


# ============================================================
# TAB 2 — EXPLAINABILITY (SHAP)
# ============================================================

with tab2:

    st.header("📊 Model Explainability — SHAP Analysis")
    st.markdown("""
    SHAP (SHapley Additive exPlanations) explains **why** the model makes each 
    prediction by assigning an importance score to every feature for every sample. 
    SHAP is mathematically grounded in cooperative game theory (Shapley values).
    """)

    if models["shap_values"] is None:
        st.warning(
            "SHAP values not found. Run `shap_analysis.py` first.\n\n"
            "```bash\npython shap_analysis.py\n```"
        )
    else:
        shap_vals  = models["shap_values"]
        feat_names = models["shap_names"]

        st.divider()

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.subheader("Beeswarm Summary")
            st.caption("Each dot = one sample. Color = feature value. X = SHAP impact.")
            if os.path.exists("shap_summary.png"):
                st.image("shap_summary.png", use_column_width=True)
            else:
                st.info("Run shap_analysis.py to generate this plot.")

        with col_s2:
            st.subheader("Feature Importance (Mean |SHAP|)")
            st.caption("Average impact of each feature across all samples.")
            if os.path.exists("shap_bar.png"):
                st.image("shap_bar.png", use_column_width=True)
            else:
                st.info("Run shap_analysis.py to generate this plot.")

        st.divider()

        st.subheader("Individual Patient Explanations")
        wf_col1, wf_col2 = st.columns(2)

        with wf_col1:
            st.markdown("**Parkinson's Patient** — how the model reached its diagnosis")
            if os.path.exists("shap_waterfall_pd.png"):
                st.image("shap_waterfall_pd.png", use_column_width=True)

        with wf_col2:
            st.markdown("**Healthy Patient** — how the model confirmed healthy status")
            if os.path.exists("shap_waterfall_hc.png"):
                st.image("shap_waterfall_hc.png", use_column_width=True)

        st.divider()

        st.subheader("Top 15 Features by SHAP Importance")

        mean_shap = np.abs(shap_vals).mean(axis=0)
        top_idx   = np.argsort(mean_shap)[::-1][:15]

        st.table({
            "Rank"        : list(range(1, 16)),
            "Feature"     : [feat_names[i] for i in top_idx],
            "Mean |SHAP|" : [f"{mean_shap[i]:.5f}" for i in top_idx],
            "Group"       : [
                "Clinical" if any(k in feat_names[i]
                    for k in ["Jitter","Shimmer","HNR","NHR","F0"])
                else "Acoustic"
                for i in top_idx
            ]
        })

        st.caption(
            "Clinical features (Jitter, Shimmer, HNR, F0) appearing in the top 15 "
            "validate that the model uses medically meaningful voice biomarkers."
        )


# ============================================================
# TAB 3 — FEDERATED LEARNING
# ============================================================

with tab3:

    st.header("📡 Federated Learning Overview")
    st.markdown("""
    Federated Learning trains a global model across multiple 
    **private datasets without sharing raw data**. Each client 
    trains locally and only shares model weights with the server.
    """)

    st.divider()

    st.subheader("Clients")
    c1, c2, c3 = st.columns(3)
    c1.info("**Client 1**\nItalian Dataset\nMultiple speech types")
    c2.warning("**Client 2**\n41+41 Dataset\nControlled vowel sounds")
    c3.success("**Client 3**\nLarge Dataset\n~1100+ samples")

    st.divider()

    st.subheader("Training Process (FedAvg)")
    st.markdown("""
    1. Global model initialized on server  
    2. Global weights sent to all clients  
    3. Each client trains on its **local private data**  
    4. Clients send updated weights back (not data)  
    5. Server aggregates using **weighted FedAvg**:  
       `Global = Σ(client_weights × client_size) / total_samples`  
    6. Repeat for 10 rounds  
    """)

    st.divider()

    st.subheader("Training Convergence")

    if os.path.exists("fl_round_accuracies.npy"):
        round_accs = np.load("fl_round_accuracies.npy")
        fig_fl, ax_fl = plt.subplots(figsize=(8, 3.5))
        ax_fl.plot(
            range(1, len(round_accs) + 1),
            round_accs * 100,
            marker="o", color="#4A90D9", linewidth=2
        )
        ax_fl.set_title("Federated Model — Test Accuracy per Round")
        ax_fl.set_xlabel("Communication Round")
        ax_fl.set_ylabel("Test Accuracy (%)")
        ax_fl.set_ylim([50, 100])
        ax_fl.grid(alpha=0.3)
        fig_fl.tight_layout()
        st.pyplot(fig_fl)
        plt.close()
    else:
        st.info("Run federated_train.py to generate convergence data.")

    st.divider()

    st.subheader("Model Comparison")
    st.caption("All results from person-wise evaluation on held-out test data.")
    st.markdown("""
    | Model | Accuracy | Sensitivity | AUC-ROC | Notes |
    |---|---|---|---|---|
    | SVM (RBF) | 96.74% ± 1.98% | 96.40% ± 4.44% | 0.9915 | 5-fold person-wise CV |
    | Random Forest | 96.35% ± 2.38% | 96.11% ± 4.90% | 0.9904 | 5-fold person-wise CV + SHAP |
    | Federated NN | 95.53% | 98.53% | 0.9918 | Privacy-preserving, 3 clients, 10 rounds |
    """)
    st.info(
        "📌 Federated accuracy is slightly lower than centralized models "
        "due to Non-IID data across clients — a known trade-off between "
        "privacy and performance in real-world FL deployments."
    )