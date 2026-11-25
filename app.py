"""Streamlit web application for Phishing & Scam Website Detection."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

from src.features import (
    FEATURE_DESCRIPTIONS,
    FEATURE_NAMES,
    URL_FEATURE_NAMES,
    analyze_url_risk,
    extract_url_features,
)

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "Dataset" / "phishing.csv"
MODEL_PATH = ROOT_DIR / "models" / "best_model.joblib"
URL_MODEL_PATH = ROOT_DIR / "models" / "url_model.joblib"
METRICS_PATH = ROOT_DIR / "models" / "metrics.json"
IMAGES_DIR = ROOT_DIR / "Images"
TRAIN_CODE_PATH = ROOT_DIR / "src" / "train.py"
FEATURES_CODE_PATH = ROOT_DIR / "src" / "features.py"

PRESET_URLS = {
    "🟢 Safe: google.com": "https://www.google.com",
    "🔴 Scam: IP Login": "http://192.168.1.1/login/verify-account",
    "🔴 Scam: Shortener + Bank": "http://bit.ly/verify-bank-login",
    "🔴 Scam: Brand Spoof": "http://paypa1-secure-login.xyz/account/verify",
    "🔴 Scam: Crypto Bonus": "http://free-crypto-airdrop-claim.tk/wallet/connect",
}

st.set_page_config(
    page_title="NetShield | Phishing & Scam Detector",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .verdict-card {
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        margin: 0.75rem 0 1rem 0;
        border-left: 6px solid;
    }
    .verdict-safe {
        background: rgba(16, 185, 129, 0.12);
        border-color: #10b981;
    }
    .verdict-phishing {
        background: rgba(239, 68, 68, 0.14);
        border-color: #ef4444;
    }
    .flag-pill {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        margin: 0.2rem 0.35rem 0.2rem 0;
        border-radius: 999px;
        font-size: 0.85rem;
        background: rgba(239, 68, 68, 0.18);
        border: 1px solid rgba(239, 68, 68, 0.45);
    }
    .safe-pill {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        margin: 0.2rem 0.35rem 0.2rem 0;
        border-radius: 999px;
        font-size: 0.85rem;
        background: rgba(16, 185, 129, 0.16);
        border: 1px solid rgba(16, 185, 129, 0.45);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_artifact():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_url_model_artifact():
    if not URL_MODEL_PATH.exists():
        return None
    return joblib.load(URL_MODEL_PATH)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH)
    if "Index" in df.columns:
        df = df.drop(columns=["Index"])
    return df


def render_prediction_page(
    artifact: dict | None,
    url_artifact: dict | None,
    df: pd.DataFrame | None,
) -> None:
    st.title("🛡️ NetShield — Phishing & Scam Detector")
    st.caption(
        "Paste any website URL or click a quick preset below to check for phishing, "
        "brand spoofing, and scam indicators."
    )

    if artifact is None:
        st.error("Trained model not found. Run `python src/train.py` first.")
        return

    mode = st.radio(
        "Choose Detection Mode",
        ["Analyze a URL", "Test Dataset Sample", "Manual 30-Feature Inspector"],
        horizontal=True,
    )

    if mode == "Analyze a URL":
        if "url_input_box" not in st.session_state:
            st.session_state["url_input_box"] = "https://www.google.com"

        st.markdown("**Quick Test Examples:**")
        preset_cols = st.columns(len(PRESET_URLS))
        for col, (label, preset_url) in zip(preset_cols, PRESET_URLS.items()):
            with col:
                if st.button(label, use_container_width=True):
                    st.session_state["url_input_box"] = preset_url

        url_input = st.text_input(
            "Website URL",
            placeholder="https://example.com/login",
            key="url_input_box",
        )

        extracted = extract_url_features(url_input) if url_input.strip() else {n: 1 for n in FEATURE_NAMES}

        if st.session_state.get("last_url") != url_input:
            st.session_state["last_url"] = url_input
            for fname in FEATURE_NAMES:
                st.session_state[f"url_{fname}"] = extracted[fname]

        features: dict[str, int] = dict(extracted)
        with st.expander("Advanced: Override Page / Domain / Traffic Features (optional)", expanded=False):
            cols = st.columns(3)
            for idx, fname in enumerate(FEATURE_NAMES):
                with cols[idx % 3]:
                    default_val = st.session_state.get(f"url_{fname}", extracted[fname])
                    features[fname] = st.selectbox(
                        fname,
                        options=[-1, 0, 1],
                        index=[-1, 0, 1].index(default_val),
                        help=FEATURE_DESCRIPTIONS.get(fname, ""),
                        key=f"url_{fname}",
                    )

        non_url_features = [f for f in FEATURE_NAMES if f not in URL_FEATURE_NAMES]
        page_customized = any(features[f] != 1 for f in non_url_features)

        if st.button("🔍 Scan Website", type="primary", use_container_width=True):
            if not url_input.strip():
                st.warning("Please enter a URL to scan.")
                return

            if page_customized or url_artifact is None:
                model = artifact["model"]
                fnames = artifact["feature_names"]
                input_df = pd.DataFrame([[features[n] for n in fnames]], columns=fnames)
                ml_phishing_prob = float(model.predict_proba(input_df)[0][0])
                model_label = f"{artifact['model_name']} (30 features)"
            else:
                model = url_artifact["model"]
                fnames = url_artifact["feature_names"]
                input_df = pd.DataFrame([[features[n] for n in fnames]], columns=fnames)
                ml_phishing_prob = float(model.predict_proba(input_df)[0][0])
                model_label = f"{url_artifact['model_name']} + Scam Heuristics"

            analysis = analyze_url_risk(url_input, ml_phishing_prob)
            risk_score = analysis["risk_score"]
            is_phishing = analysis["is_phishing"]

            if is_phishing:
                st.markdown(
                    f"""
                    <div class="verdict-card verdict-phishing">
                        <h3 style="margin:0;color:#f87171;">🚨 Phishing / Scam Website Detected</h3>
                        <p style="margin:0.4rem 0 0 0;">
                            <strong>Risk Score:</strong> {risk_score:.1f}% &nbsp;|&nbsp;
                            <strong>Engine:</strong> {model_label}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="verdict-card verdict-safe">
                        <h3 style="margin:0;color:#34d399;">✅ Legitimate Website</h3>
                        <p style="margin:0.4rem 0 0 0;">
                            <strong>Safety Confidence:</strong> {100 - risk_score:.1f}% &nbsp;|&nbsp;
                            <strong>Engine:</strong> {model_label}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.progress(min(max(int(round(risk_score)), 0), 100), text=f"Phishing Risk Meter: {risk_score:.1f}%")

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("⚠️ Risk Signals")
                if analysis["red_flags"]:
                    pills = "".join(f'<span class="flag-pill">{flag}</span>' for flag in analysis["red_flags"])
                    st.markdown(pills, unsafe_allow_html=True)
                else:
                    st.caption("No suspicious URL or scam indicators detected.")
            with c2:
                st.subheader("🛡️ Safety Checks")
                if analysis["safe_signals"]:
                    pills = "".join(f'<span class="safe-pill">{sig}</span>' for sig in analysis["safe_signals"])
                    st.markdown(pills, unsafe_allow_html=True)
                else:
                    st.caption("No verified safety signals.")

            with st.expander("View Extracted Feature Vector"):
                st.dataframe(input_df, use_container_width=True)

    elif mode == "Test Dataset Sample":
        if df is None:
            st.warning("Dataset not found.")
            return
        row_idx = st.number_input("Dataset Row Index", min_value=0, max_value=len(df) - 1, value=0)
        row = df.iloc[int(row_idx)]
        actual_label = "Legitimate (1)" if int(row["class"]) == 1 else "Phishing (-1)"
        st.info(f"**Actual Label in Dataset:** `{actual_label}` | **Model:** {artifact['model_name']} (30 features)")

        if st.button("🔍 Predict Dataset Sample", type="primary", use_container_width=True):
            fnames = artifact["feature_names"]
            input_df = pd.DataFrame([[int(row[n]) for n in fnames]], columns=fnames)
            pred = int(artifact["model"].predict(input_df)[0])
            proba = artifact["model"].predict_proba(input_df)[0]
            if pred == 1:
                st.success(f"✅ **Legitimate Website** ({proba[1] * 100:.2f}% confidence)")
            else:
                st.error(f"🚨 **Phishing Website Detected** ({proba[0] * 100:.2f}% confidence)")
            with st.expander("Sample Feature Vector"):
                st.dataframe(input_df, use_container_width=True)

    else:
        st.info(f"**Model:** {artifact['model_name']} (30 features)")
        features = {}
        cols = st.columns(3)
        for idx, fname in enumerate(FEATURE_NAMES):
            with cols[idx % 3]:
                features[fname] = st.selectbox(
                    fname,
                    options=[-1, 0, 1],
                    index=2,
                    help=FEATURE_DESCRIPTIONS.get(fname, ""),
                    key=f"manual_{fname}",
                )
        if st.button("🔍 Predict Manual Features", type="primary", use_container_width=True):
            fnames = artifact["feature_names"]
            input_df = pd.DataFrame([[features[n] for n in fnames]], columns=fnames)
            pred = int(artifact["model"].predict(input_df)[0])
            proba = artifact["model"].predict_proba(input_df)[0]
            if pred == 1:
                st.success(f"✅ **Legitimate Website** ({proba[1] * 100:.2f}% confidence)")
            else:
                st.error(f"🚨 **Phishing Website Detected** ({proba[0] * 100:.2f}% confidence)")


def render_comparison_page(metrics: dict | None) -> None:
    st.title("📊 Model Comparison & Evaluation")
    if metrics is None:
        st.error("Metrics file not found. Run `python src/train.py` first.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Rows", f"{metrics['dataset_rows']:,}")
    c2.metric("Features", metrics["feature_count"])
    c3.metric("Best Model", metrics["best_model"])
    c4.metric("Best Accuracy", f"{metrics['best_accuracy'] * 100:.2f}%")

    table_rows = [
        {
            "Model": r["model"],
            "Accuracy (%)": round(r["accuracy"] * 100, 2),
            "Precision (%)": round(r["precision"] * 100, 2),
            "Recall (%)": round(r["recall"] * 100, 2),
            "F1 Score (%)": round(r["f1_score"] * 100, 2),
        }
        for r in metrics["results"]
    ]
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    comp_img = IMAGES_DIR / "model_comparison.png"
    if comp_img.exists():
        st.image(str(comp_img), caption="Accuracy Comparison Across 8 Classifiers")

    st.subheader("Confusion Matrices")
    selected_model = st.selectbox(
        "Select Classifier",
        [r["model"] for r in metrics["results"]],
    )
    selected_row = next(r for r in metrics["results"] if r["model"] == selected_model)
    cm_path = ROOT_DIR / selected_row["image"]
    if cm_path.exists():
        st.image(str(cm_path), width=460)


def render_dataset_page(df: pd.DataFrame | None) -> None:
    st.title("📁 Dataset Explorer")
    if df is None:
        st.error("Dataset/phishing.csv not found.")
        return

    counts = df["class"].value_counts().rename(index={1: "Legitimate (1)", -1: "Phishing (-1)"})
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Samples", f"{len(df):,}")
    c2.metric("Legitimate Sites", f"{int(counts.get('Legitimate (1)', 0)):,}")
    c3.metric("Phishing Sites", f"{int(counts.get('Phishing (-1)', 0)):,}")

    st.bar_chart(counts)
    st.subheader("Sample Records")
    st.dataframe(df.head(50), use_container_width=True)


def render_code_page() -> None:
    st.title("💻 Model & Feature Extraction Code")
    st.caption("Inspect the full training pipeline (`src/train.py`) and URL scam/feature analyzer (`src/features.py`).")

    tab1, tab2 = st.tabs(["Training Pipeline (src/train.py)", "Scam & Feature Analyzer (src/features.py)"])
    with tab1:
        if TRAIN_CODE_PATH.exists():
            st.code(TRAIN_CODE_PATH.read_text(encoding="utf-8"), language="python")
        else:
            st.warning("src/train.py not found.")
    with tab2:
        if FEATURES_CODE_PATH.exists():
            st.code(FEATURES_CODE_PATH.read_text(encoding="utf-8"), language="python")
        else:
            st.warning("src/features.py not found.")


def main() -> None:
    with st.sidebar:
        selected = option_menu(
            "NetShield",
            ["Detect Website", "Model Comparison", "Dataset Explorer", "Model Code"],
            icons=["shield-check", "bar-chart", "table", "code-slash"],
            menu_icon="shield-lock",
            default_index=0,
        )

    artifact = load_model_artifact()
    url_artifact = load_url_model_artifact()
    metrics = load_metrics()
    df = load_dataset()

    if selected == "Detect Website":
        render_prediction_page(artifact, url_artifact, df)
    elif selected == "Model Comparison":
        render_comparison_page(metrics)
    elif selected == "Dataset Explorer":
        render_dataset_page(df)
    else:
        render_code_page()


if __name__ == "__main__":
    main()
