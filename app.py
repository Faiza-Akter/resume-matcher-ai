import streamlit as st
import joblib
import numpy as np
import json

from pathlib import Path
from src.config import CONFIG
from src.preprocess import clean_text
from src.resume_parser import extract_text


# -------------------------
# Load trained model files
# -------------------------
@st.cache_resource
def load_model_artifacts():
    if not CONFIG.vectorizer_path.exists() or not CONFIG.classifier_path.exists():
        st.error("❌ Model not found. Please train the model first.")
        return None, None, None

    vectorizer = joblib.load(CONFIG.vectorizer_path)
    classifier = joblib.load(CONFIG.classifier_path)

    # Load accuracy and metrics
    if CONFIG.label_info_path.exists():
        with open(CONFIG.label_info_path, "r") as f:
            label_info = json.load(f)
        metrics = label_info.get("metrics", {})
    else:
        metrics = {}

    return vectorizer, classifier, metrics


vectorizer, classifier, metrics = load_model_artifacts()


# -------------------------
# STREAMLIT UI
# -------------------------
st.set_page_config(page_title="AI Resume Matcher", page_icon="🤖", layout="wide")
st.title("🤖 AI Resume Matcher using NLP + Machine Learning")
st.write("Upload resumes and compare them against any job description.")

# -------------------------
# Job Description Input
# -------------------------
st.subheader("📌 Step 1: Enter Job Description")
job_description = st.text_area(
    "Paste the job description below:",
    height=180,
    placeholder="Example: Looking for a Python Developer with experience in ML, APIs, SQL..."
)


# -------------------------
# Resume Upload
# -------------------------
st.subheader("📌 Step 2: Upload Resumes (PDF, DOCX, TXT)")
uploaded_files = st.file_uploader(
    "Upload multiple resumes:",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)


# -------------------------
# Match Function
# -------------------------
def compute_match_score(job_text, resume_text):
    job_clean = clean_text(job_text)
    resume_clean = clean_text(resume_text)

    combined = f"JOB: {job_clean} RESUME: {resume_clean}"

    X = vectorizer.transform([combined])
    pred = classifier.predict_proba(X)[0][1]  # probability of class 1 = good match

    return float(pred * 100)  # convert to percentage


# -------------------------
# Display Model Accuracy
# -------------------------
if metrics:
    st.info(f"📊 **Model Accuracy:** {metrics.get('accuracy', 'N/A')}%")
else:
    st.warning("⚠ Model metrics not found. Train the model to get accuracy stats.")


# -------------------------
# Submit Button
# -------------------------
if st.button("🔍 Match Resumes"):
    if not job_description.strip():
        st.error("❌ Please enter a job description.")
    elif not uploaded_files:
        st.error("❌ Please upload at least one resume.")
    else:
        st.success("Processing resumes...")

        results = []

        for file in uploaded_files:
            file_bytes = file.read()
            resume_text = extract_text(file.name, file_bytes)

            if not resume_text.strip():
                st.warning(f"⚠ Could not read text from file: {file.name}")
                continue

            score = compute_match_score(job_description, resume_text)

            if score >= 80:
                category = "🟢 Strong Match"
            elif score >= 50:
                category = "🟡 Moderate Match"
            else:
                category = "🔴 Weak Match"

            results.append((file.name, score, category))

        st.subheader("📌 Match Results")

        for name, score, category in results:
            st.write(f"### 📄 {name}")
            st.write(f"**Match Score:** {score:.2f}%")
            st.write(f"**Category:** {category}")
            st.write("---")
