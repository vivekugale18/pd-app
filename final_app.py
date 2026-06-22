import streamlit as st
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
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
    page_title="Parkinson's Early Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #F8FAFC, #EEF4FF);
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    color: #1E3A8A;
    margin-bottom: 0px;
}

.subtitle {
    text-align: center;
    color: #64748B;
    font-size: 18px;
    margin-bottom: 20px;
}

div[data-testid="metric-container"] {
    background: white;
    border-radius: 14px;
    padding: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    border: 1px solid #E2E8F0;
}

.stTabs [data-baseweb="tab"] {
    font-size: 18px;
    font-weight: 700;
}

.custom-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 5px 18px rgba(0,0,0,0.08);
    border: 1px solid #E2E8F0;
}

.hero-blue {
    background: linear-gradient(90deg,#2563EB,#1D4ED8);
    padding: 20px;
    border-radius: 14px;
    color: white;
}

.hero-purple {
    background: linear-gradient(90deg,#7C3AED,#4F46E5);
    padding: 20px;
    border-radius: 14px;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    models = {}

    if os.path.exists("imputer.pkl"):
        models["imputer"] = joblib.load("imputer.pkl")
    else:
        models["imputer"] = SimpleImputer(strategy="median")
        models["imputer"].fit(np.zeros((1, TOTAL_FEATURES)))

    if os.path.exists("svm_model.pkl") and os.path.exists("scaler.pkl"):
        models["svm"] = joblib.load("svm_model.pkl")
        models["scaler"] = joblib.load("scaler.pkl")
    else:
        models["svm"] = None
        models["scaler"] = None

    if os.path.exists("rf_model.pkl"):
        models["rf"] = joblib.load("rf_model.pkl")
    else:
        models["rf"] = None

    if os.path.exists("federated_model.pth"):
        fl = ParkinsonNN(input_size=TOTAL_FEATURES)
        fl.load_state_dict(torch.load("federated_model.pth", map_location="cpu"))
        fl.eval()
        models["fl"] = fl
    else:
        models["fl"] = None

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

    if os.path.exists("shap_values.npy") and os.path.exists("shap_feature_names.npy"):
        models["shap_values"] = np.load("shap_values.npy", allow_pickle=True)
        models["shap_names"] = np.load(
            "shap_feature_names.npy",
            allow_pickle=True
        ).tolist()

        if models["rf"] is not None:
            models["shap_explainer"] = shap.TreeExplainer(models["rf"])
        else:
            models["shap_explainer"] = None
    else:
        models["shap_values"] = None
        models["shap_names"] = None
        models["shap_explainer"] = None

    return models


models = load_models()

# ============================================================
# SHAP HELPER
# ============================================================

def get_shap_explanation(feat_scaled):
    explainer = models["shap_explainer"]

    if explainer is None:
        return None, None

    try:
        explanation = explainer(feat_scaled)

        if explanation.values.ndim == 3:
            vals = explanation.values[0, :, 1]
            base = float(explanation.base_values[0, 1])

        elif explanation.values.ndim == 2:
            vals = explanation.values[0, :]
            base_raw = explanation.base_values
            base = (
                float(base_raw[0])
                if hasattr(base_raw, '__len__')
                else float(base_raw)
            )
        else:
            vals = explanation.values
            base = float(explanation.base_values)

        return vals, base

    except Exception:
        sv = explainer.shap_values(feat_scaled)
        ev = explainer.expected_value

        if isinstance(sv, list) and len(sv) == 2:
            sv_class1 = sv[1]
            ev_class1 = ev[1] if hasattr(ev, '__len__') else float(ev)

        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            sv_class1 = sv[:, :, 1]
            ev_class1 = ev[1] if hasattr(ev, '__len__') else float(ev)

        else:
            sv_class1 = sv
            ev_class1 = (
                ev[1]
                if (hasattr(ev, '__len__') and len(ev) >= 2)
                else float(ev)
            )

        if isinstance(sv_class1, np.ndarray) and sv_class1.ndim == 2:
            vals = sv_class1[0]
        elif isinstance(sv_class1, np.ndarray) and sv_class1.ndim == 1:
            vals = sv_class1
        else:
            vals = np.array(sv_class1).flatten()

        return vals, float(ev_class1)


# ============================================================
# VISUAL HELPERS
# ============================================================

def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(8, 3))
    time = np.linspace(0, len(y) / sr, len(y))
    ax.plot(time, y, color="#2563EB", linewidth=0.8)
    ax.set_title("Audio Waveform", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_spectrogram(y, sr):
    fig, ax = plt.subplots(figsize=(8, 3))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    img = ax.imshow(
        S_db,
        aspect="auto",
        origin="lower",
        cmap="magma",
        interpolation="nearest"
    )

    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel Spectrogram", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Mel Bands")
    fig.tight_layout()
    return fig


# ============================================================
# PREDICTION HELPERS
# ============================================================

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
    raw features -> imputer -> scaler -> model
    returns: pred, prob, prob_pd
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

    else:
        if models["fl"] is None:
            return None, None, None

        feat = models["fl_imputer"].transform([features_raw])
        feat = models["fl_scaler"].transform(feat)

        tensor = torch.tensor(feat, dtype=torch.float32)

        with torch.no_grad():
            out = models["fl"](tensor)
            prob = torch.softmax(out, dim=1).numpy()[0]
            pred = int(np.argmax(prob))

    return int(pred), prob, float(prob[1])


# ============================================================
# TITLE + TABS
# ============================================================

st.markdown(
    '<div class="main-title">🧠 Parkinson\'s Early Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Voice-based Detection using Machine Learning • Federated Learning • Explainable AI</div>',
    unsafe_allow_html=True
)

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
        st.markdown("""
        <div class='custom-card'>
        <h3>⚙ Detection Settings</h3>
        <p>Select model and upload voice sample</p>
        </div>
        """, unsafe_allow_html=True)

        model_choice = st.radio(
            "Select Model",
            ["SVM", "Random Forest", "Federated Model"],
            help=(
                "SVM and Random Forest use centralized training. "
                "Federated model preserves privacy using distributed training."
            )
        )

        uploaded_file = st.file_uploader(
            "Upload Voice Recording (.wav)",
            type=["wav"],
            help="Upload sustained vowel 'ahh' recording (3–10 seconds)"
        )

    with col_right:

        if uploaded_file is None:
            st.info("👈 Upload a WAV file to begin analysis")

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
                st.error("Audio too short — upload at least 3 seconds.")

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
                    st.error("Model not loaded. Run training first.")

                else:
                    avg_prob = np.mean(all_probs, axis=0)
                    final_pred = int(np.argmax(avg_prob))
                    confidence = float(min(np.max(avg_prob), 0.99))
                    risk, color = risk_level(confidence, final_pred)

                    st.divider()
                    st.subheader("Diagnostic Result")

                    if final_pred == 1:
                        st.markdown("""
                        <div style="
                            background:#FEF2F2;
                            padding:25px;
                            border-left:8px solid #EF4444;
                            border-radius:14px;">
                            <h2 style="color:#DC2626;">⚠ Parkinson's Risk Detected</h2>
                            <p>Voice biomarkers indicate possible Parkinson's patterns.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="
                            background:#ECFDF5;
                            padding:25px;
                            border-left:8px solid #10B981;
                            border-radius:14px;">
                            <h2 style="color:#059669;">✅ Healthy Voice Pattern</h2>
                            <p>No strong Parkinson's biomarkers detected.</p>
                        </div>
                        """, unsafe_allow_html=True)

                    st.write("")

                    m1, m2, m3 = st.columns(3)

                    with m1:
                        st.metric("Confidence", f"{confidence*100:.1f}%")

                    with m2:
                        st.metric("Risk Level", risk)

                    with m3:
                        st.metric("Segments", len(segments))

                    st.caption(f"Model Used: {model_choice}")

                    with st.expander("Segment-Level Breakdown"):
                        st.table({
                            "Segment": [f"Seg {i+1}" for i in range(len(all_preds))],
                            "Prediction": [
                                "Parkinson's" if p == 1 else "Healthy"
                                for p in all_preds
                            ],
                            "PD Probability": [
                                f"{p[1]*100:.1f}%"
                                for p in all_probs
                            ]
                        })

                    if model_choice == "Random Forest" and models["shap_explainer"] is not None:
                        with st.expander("🔍 SHAP Explanation — Why this prediction?"):
                            st.info("""
                            🔴 Red features push toward Parkinson's  
                            🔵 Blue features push toward Healthy  
                            Higher magnitude = stronger contribution
                            """)

                            try:
                                feat_imp = models["imputer"].transform([all_feats[0]])
                                feat_scaled = models["scaler"].transform(feat_imp)

                                sv_pd, ev_pd = get_shap_explanation(feat_scaled)

                                if sv_pd is not None:
                                    plt.rcParams.update({'font.size': 11})
                                    fig_wf, _ = plt.subplots(figsize=(12, 7), dpi=130)

                                    shap.waterfall_plot(
                                        shap.Explanation(
                                            values=sv_pd,
                                            base_values=ev_pd,
                                            data=feat_scaled[0],
                                            feature_names=models["shap_names"]
                                        ),
                                        max_display=12,
                                        show=False
                                    )

                                    plt.tight_layout()
                                    st.pyplot(fig_wf)
                                    plt.close()

                            except Exception as e:
                                st.warning(f"SHAP unavailable: {e}")

                    elif model_choice == "SVM":
                        with st.expander("ℹ SHAP for SVM"):
                            st.info(
                                "Live SHAP explanation is only available for Random Forest. "
                                "Use Explainability tab for global feature analysis."
                            )

                    elif model_choice == "Federated Model":
                        with st.expander("ℹ SHAP for Federated Model"):
                            st.info(
                                "Per-prediction SHAP not available for Federated model. "
                                "See global SHAP in Explainability tab."
                            )

# ============================================================
# TAB 2 — EXPLAINABILITY (SHAP)
# ============================================================

with tab2:

    st.markdown("""
    <div class='hero-blue'>
    <h2>📊 Explainability Dashboard</h2>
    <p>Understand why the model makes a prediction using SHAP values</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.info("""
    SHAP (SHapley Additive exPlanations) explains model decisions using game theory.

    🔴 Red = pushes prediction toward Parkinson's  
    🔵 Blue = pushes prediction toward Healthy  
    Larger SHAP value = stronger feature influence
    """)

    if models["shap_values"] is None:
        st.warning(
            "SHAP values not found. Run:\n\n"
            "python shap_analysis.py"
        )

    else:
        shap_vals = models["shap_values"]
        feat_names = models["shap_names"]

        st.divider()

        col_s1, col_s2 = st.columns(2)

        with col_s1:
            st.markdown("""
            <div class='custom-card'>
            <h4>Beeswarm Summary Plot</h4>
            <p>Each dot represents one sample</p>
            </div>
            """, unsafe_allow_html=True)

            if os.path.exists("shap_summary.png"):
                st.image("shap_summary.png", use_container_width=True)
            else:
                st.info("Run shap_analysis.py to generate summary plot.")

        with col_s2:
            st.markdown("""
            <div class='custom-card'>
            <h4>Feature Importance Plot</h4>
            <p>Mean absolute SHAP importance</p>
            </div>
            """, unsafe_allow_html=True)

            if os.path.exists("shap_bar.png"):
                st.image("shap_bar.png", use_container_width=True)
            else:
                st.info("Run shap_analysis.py to generate bar plot.")

        st.divider()

        st.subheader("Individual Patient Explanations")

        wf_col1, wf_col2 = st.columns(2)

        with wf_col1:
            st.markdown("""
            <div class='custom-card'>
            <h4>Parkinson's Patient</h4>
            <p>Why model predicted PD</p>
            </div>
            """, unsafe_allow_html=True)

            if os.path.exists("shap_waterfall_pd.png"):
                st.image("shap_waterfall_pd.png", use_container_width=True)
            else:
                st.info("Waterfall plot unavailable.")

        with wf_col2:
            st.markdown("""
            <div class='custom-card'>
            <h4>Healthy Patient</h4>
            <p>Why model predicted Healthy</p>
            </div>
            """, unsafe_allow_html=True)

            if os.path.exists("shap_waterfall_hc.png"):
                st.image("shap_waterfall_hc.png", use_container_width=True)
            else:
                st.info("Waterfall plot unavailable.")

        st.divider()

        st.subheader("Top 15 Most Important Features")

        mean_shap = np.abs(shap_vals).mean(axis=0)
        top_idx = np.argsort(mean_shap)[::-1][:15]

        st.table({
            "Rank": list(range(1, 16)),
            "Feature": [feat_names[i] for i in top_idx],
            "Mean |SHAP|": [f"{mean_shap[i]:.5f}" for i in top_idx],
            "Feature Type": [
                "Clinical"
                if any(
                    k in feat_names[i]
                    for k in ["Jitter", "Shimmer", "HNR", "NHR", "F0"]
                )
                else "Acoustic"
                for i in top_idx
            ]
        })

        st.success(
            "Clinical biomarkers like Jitter, Shimmer, HNR, and F0 appearing "
            "among top features validate the medical relevance of the model."
        )
# ============================================================
# TAB 3 — FEDERATED LEARNING
# ============================================================

with tab3:

    st.markdown("""
    <div class='hero-purple'>
    <h2>📡 Federated Learning Dashboard</h2>
    <p>Privacy-preserving collaborative model training across institutions</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.markdown("""
    Federated Learning enables multiple hospitals or research centers to train a shared
    AI model **without exchanging raw patient voice data**.

    Instead of sending audio data to a central server:
    - Each client trains locally
    - Only model weights are shared
    - Server aggregates weights using **FedAvg**
    """)

    st.divider()

    st.subheader("Participating Clients")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class='custom-card'>
        <h4>🏥 Client 1</h4>
        <p>Italian Dataset</p>
        <h2>46</h2>
        <p>Patients</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class='custom-card'>
        <h4>🏥 Client 2</h4>
        <p>Controlled Vowel Dataset</p>
        <h2>82</h2>
        <p>Patients</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class='custom-card'>
        <h4>🏥 Client 3</h4>
        <p>Large Multi-source Dataset</p>
        <h2>610+</h2>
        <p>Patients</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.subheader("Training Pipeline (FedAvg)")

    st.markdown("""
    **Step 1:** Initialize global model on server  
    **Step 2:** Send model weights to all clients  
    **Step 3:** Each client trains on local private data  
    **Step 4:** Clients send updated weights back  
    **Step 5:** Server aggregates using weighted FedAvg  
    **Step 6:** Repeat for 10 communication rounds  
    """)

    st.divider()

    st.subheader("Performance Metrics")

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric("Final Accuracy", "95.53%")

    with m2:
        st.metric("Sensitivity", "98.53%")

    with m3:
        st.metric("Communication Rounds", "10")

    st.divider()

    st.subheader("Training Convergence")

    if os.path.exists("fl_round_accuracies.npy"):
        round_accs = np.load("fl_round_accuracies.npy") * 100
        rounds = list(range(1, len(round_accs) + 1))

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=rounds,
            y=round_accs,
            mode='lines+markers',
            name='Global Accuracy',
            line=dict(width=4, color='royalblue'),
            marker=dict(size=10)
        ))

        fig.update_layout(
            title="Federated Training Convergence",
            xaxis_title="Communication Round",
            yaxis_title="Accuracy (%)",
            template="plotly_white",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Run federated_train.py to generate convergence data.")

    st.divider()

    st.subheader("Model Comparison")

    st.markdown("""
    | Model | Accuracy | Sensitivity | AUC-ROC | Notes |
    |---|---|---|---|---|
    | SVM (RBF) | 96.74% | 96.40% | 0.9915 | Centralized |
    | Random Forest | 96.35% | 96.11% | 0.9904 | SHAP Enabled |
    | Federated NN | 95.53% | 98.53% | 0.9918 | Privacy Preserving |
    """)

    st.info(
        "📌 Federated Learning shows slightly lower accuracy than centralized "
        "models due to Non-IID client data distribution, but provides strong "
        "privacy guarantees for healthcare AI deployment."
    )