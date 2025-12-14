import base64
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import CONFIG
from src.preprocess import build_pair_text, clean_text
from src.resume_parser import extract_text


# -------------------------
# Page Config (NOT wide)
# -------------------------
st.set_page_config(page_title="AI Resume Matcher", page_icon="📄", layout="centered")


# -------------------------
# Background Image + Layout CSS
# -------------------------
def _load_bg_base64(img_path: str):
    p = Path(img_path)
    if not p.exists():
        return None, None
    ext = p.suffix.lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"
    if ext not in {"jpeg", "png", "webp"}:
        ext = "jpeg"
    return base64.b64encode(p.read_bytes()).decode("utf-8"), ext


bg_b64, bg_ext = _load_bg_base64("assets/bggg.png")

bg_css = ""
if bg_b64:
    bg_css = f"""
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/{bg_ext};base64,{bg_b64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    """

st.markdown(
    f"""
    <style>
    {bg_css}

    /* Remove the blurred/white floating rectangle in the top area */
    [data-testid="stHeader"] {{
        display: none;
    }}
    [data-testid="stToolbar"] {{
        display: none;
    }}
    [data-testid="stDecoration"] {{
        display: none;
    }}

    /* Make content scroll inside the page */
    html, body {{
        height: 100%;
        overflow: hidden;
    }}
    [data-testid="stAppViewContainer"] {{
        height: 100vh;
        overflow: hidden;
    }}
    section.main {{
        height: 100vh;
        overflow-y: auto;
        padding-bottom: 2rem;
    }}

    /* Narrow the content + center it (fix too wide) */
    div.block-container {{
        max-width: 860px;
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }}

    /* Soft panel (NO blur, very light) */
    .soft-panel {{
        background: rgba(255, 255, 255, 0.10);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 18px;
        padding: 18px;
    }}

    /* Title styling (CENTERED) */
    .app-title {{
        font-size: 46px;
        font-weight: 900;
        letter-spacing: -0.5px;
        color: #0b1678;
        margin: 0.2rem 0 0.2rem 0;
        text-shadow: 0 8px 24px rgba(11, 22, 120, 0.15);
        text-align: center;
        width: 100%;
    }}
    .app-subtitle {{
        color: rgba(10, 10, 20, 0.85);
        font-size: 16px;
        margin-bottom: 0.9rem;
        text-align: center;
        width: 100%;
    }}

    /* Status cards */
    .status-card {{
        border-radius: 14px;
        padding: 14px 16px;
        font-weight: 700;
        box-shadow: 0 10px 22px rgba(0,0,0,0.10);
        border: 1px solid rgba(255,255,255,0.18);
        height: 100%;
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    .status-primary {{
        background: #0b1678;
        color: white;
    }}
    .status-secondary {{
        background: #8d97c1;
        color: #0b1678;
    }}
    .status-icon {{
        font-size: 18px;
        line-height: 1;
    }}
    .status-text {{
        font-size: 15px;
        line-height: 1.25;
    }}

    /* Center button container (ALWAYS centered) */
    .center-btn {{
        width: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 1.1rem 0 1.2rem 0;
    }}

    /* Match button gradient: #0b1678 -> #8d97c1 (BOLD text) */
    div.stButton > button {{
        border: 0 !important;
        color: white !important;
        font-weight: 900 !important;
        padding: 0.95rem 2.1rem !important;
        border-radius: 16px !important;
        background: linear-gradient(90deg, #0b1678, #8d97c1) !important;
        box-shadow: 0 14px 30px rgba(11, 22, 120, 0.28) !important;
        transition: transform 0.08s ease-in-out, box-shadow 0.08s ease-in-out;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
    }}
    div.stButton {{
        width: 100%;
        display: flex;
        justify-content: center;
    }}
    div.stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 16px 34px rgba(11, 22, 120, 0.34) !important;
    }}
    div.stButton > button:active {{
        transform: translateY(0px);
        box-shadow: 0 11px 22px rgba(11, 22, 120, 0.22) !important;
    }}

    /* File uploader "Browse files" button color */
    [data-testid="stFileUploader"] button {{
        background: #8d97c1 !important;
        color: #0b1678 !important;
        border: 0 !important;
        font-weight: 800 !important;
        border-radius: 12px !important;
        box-shadow: 0 10px 18px rgba(141, 151, 193, 0.25) !important;
    }}
    [data-testid="stFileUploader"] button:hover {{
        filter: brightness(0.98);
    }}

    /* -------------------------
       RESULT TABLE HEADER COLORS
       Header background: #0b1678
       Header text: #8d97c1
    ------------------------- */
    [data-testid="stDataFrame"] thead tr th {{
        background: #0b1678 !important;
        color: #8d97c1 !important;
        font-weight: 900 !important;
        border-bottom: 1px solid rgba(255,255,255,0.22) !important;
    }}
    [data-testid="stDataFrame"] thead tr th:first-child {{
        background: #0b1678 !important;
        color: #8d97c1 !important;
    }}
    [data-testid="stDataFrame"] tbody tr td {{
        background: rgba(255,255,255,0.92) !important;
        color: #0b1678 !important;
        border-bottom: 1px solid rgba(11,22,120,0.08) !important;
    }}
    [data-testid="stDataFrame"] {{
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.22);
        box-shadow: 0 12px 26px rgba(0,0,0,0.10);
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------
# Load trained model files
# -------------------------
@st.cache_resource
def load_model_artifacts():
    if not CONFIG.vectorizer_path.exists() or not CONFIG.classifier_path.exists():
        return None, None, {}

    vectorizer = joblib.load(CONFIG.vectorizer_path)
    classifier = joblib.load(CONFIG.classifier_path)

    metrics = {}
    if CONFIG.label_info_path.exists():
        try:
            with open(CONFIG.label_info_path, "r", encoding="utf-8") as f:
                label_info = json.load(f)
            metrics = label_info.get("metrics", {}) or {}
        except Exception:
            metrics = {}

    return vectorizer, classifier, metrics


vectorizer, classifier, metrics = load_model_artifacts()


# -------------------------
# ML Probability (Classifier)
# -------------------------
def compute_ml_probability(job_text: str, resume_text: str) -> float | None:
    """
    Returns probability of class 1 (match) from the trained model.
    If model is not available, returns None.
    """
    if vectorizer is None or classifier is None:
        return None

    combined = build_pair_text(job_text, resume_text)
    X = vectorizer.transform([combined])

    if hasattr(classifier, "predict_proba"):
        return float(classifier.predict_proba(X)[0][1])

    pred = int(classifier.predict(X)[0])
    return 1.0 if pred == 1 else 0.0


# -------------------------
# UI
# -------------------------
st.markdown('<div class="soft-panel">', unsafe_allow_html=True)

st.markdown('<div class="app-title">AI Resume Matcher</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Upload resumes and compare them against any job description.</div>',
    unsafe_allow_html=True,
)

c1, c2 = st.columns(2)

with c1:
    if vectorizer is None or classifier is None:
        st.markdown(
            """
            <div class="status-card status-secondary">
                <div class="status-icon">⚠️</div>
                <div class="status-text">Model not found. Train first: <code>py -u -m src.model --train</code></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="status-card status-primary">
                <div class="status-icon">✅</div>
                <div class="status-text">Model loaded successfully.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with c2:
    acc = metrics.get("accuracy", "N/A") if metrics else "N/A"
    st.markdown(
        f"""
        <div class="status-card status-secondary">
            <div class="status-icon">📊</div>
            <div class="status-text">Training Accuracy (dataset): {acc}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.subheader("📌 Step 1: Enter Job Description")
job_description = st.text_area(
    "Paste the job description below:",
    height=180,
    placeholder="Example: Looking for a Python Developer with experience in ML, NLP, APIs, SQL..."
)

st.subheader("📌 Step 2: Upload Resumes (PDF, DOCX, TXT)")
uploaded_files = st.file_uploader(
    "Upload multiple resumes:",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)


# -------------------------
# Similarity (internal)
# -------------------------
def compute_similarity_scores(job_text: str, resume_texts: list[str]) -> list[float]:
    job_clean = clean_text(job_text)
    resumes_clean = [clean_text(t) for t in resume_texts]

    corpus = [job_clean] + resumes_clean
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    mat = tfidf.fit_transform(corpus)

    sims = cosine_similarity(mat[0:1], mat[1:]).flatten()
    return [float(x) for x in sims]


def similarity_category(sim_percent: float) -> str:
    if sim_percent >= 35:
        return "🟢 Strong Match"
    if sim_percent >= 20:
        return "🟡 Moderate Match"
    return "🔴 Weak Match"


# -------------------------
# Centered button (CENTERED + BOLD)
# -------------------------
st.markdown('<div class="center-btn">', unsafe_allow_html=True)
match_clicked = st.button("Match Resumes")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# -------------------------
# Results
# -------------------------
if match_clicked:
    if not job_description.strip():
        st.error("❌ Please enter a job description.")
        st.stop()

    if not uploaded_files:
        st.error("❌ Please upload at least one resume.")
        st.stop()

    resume_names, resume_texts = [], []
    for file in uploaded_files:
        text = extract_text(file.name, file.read())
        if not text.strip():
            st.warning(f"⚠ Could not read text from file: {file.name}")
            continue
        resume_names.append(file.name)
        resume_texts.append(text)

    if not resume_texts:
        st.error("❌ No readable resumes found. Try TXT/DOCX or a text-based PDF.")
        st.stop()

    sims = compute_similarity_scores(job_description, resume_texts)
    sims_percent = [s * 100.0 for s in sims]

    rows = []
    for name, raw_text, sim_p in zip(resume_names, resume_texts, sims_percent):
        ml_prob = compute_ml_probability(job_description, raw_text)
        prob_percent = round(ml_prob * 100.0, 2) if ml_prob is not None else None

        rows.append({
            "Resume": name,
            "Category": similarity_category(sim_p),
            "Probability(%)": prob_percent,
            "_sim": sim_p,
        })

    df = pd.DataFrame(rows).sort_values("_sim", ascending=False).reset_index(drop=True)

    st.subheader("📌 Match Results (Ranked)")
    st.dataframe(df[["Resume", "Category", "Probability(%)"]], use_container_width=True)
