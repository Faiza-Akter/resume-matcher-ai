from typing import Tuple, Optional, Dict, Any
import pandas as pd

from .config import CONFIG
from .utils import safe_str
from .preprocess import build_pair_text

TEXT_COL_CANDIDATES = {
    "resume": [
        "resume", "resume_text", "Resume", "ResumeText",
        "cv", "cv_text", "candidate_resume"
    ],
    "job": [
        "job", "job_description", "job_desc", "JobDescription",
        "jd", "jd_text", "description"
    ],
}

LABEL_COL_CANDIDATES = ["label", "match", "is_match", "y", "target"]
SCORE_COL_CANDIDATES = ["score", "match_score", "similarity", "rating"]


def _find_col(df: pd.DataFrame, candidates) -> Optional[str]:
    cols = set(df.columns)

    for c in candidates:
        if c in cols:
            return c

    lower_map = {x.lower(): x for x in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    return None


def load_hf_dataframe(dataset_name: str = CONFIG.hf_dataset_name) -> pd.DataFrame:
    import json
    from huggingface_hub import list_repo_files, hf_hub_download

    repo_id = dataset_name
    files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    json_files = [f for f in files if f.lower().endswith(".json")]

    print(f"[HF] Found {len(json_files)} JSON files. Downloading/parsing...")

    rows = []
    total = len(json_files)

    for i, fname in enumerate(json_files, start=1):
        low = fname.lower()

        if "invalid" in low:
            continue

        if "mismatch" in low or "mismatched" in low:
            label = 0
        elif "match" in low:
            label = 1
        else:
            continue

        if i % 50 == 0:
            print(f"[HF] Processed {i}/{total} files...")

        local_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=fname)
        with open(local_path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        inp = obj.get("input", {}) if isinstance(obj, dict) else {}
        job_text = safe_str(inp.get("job_description", ""))
        resume_text = safe_str(inp.get("resume", ""))

        rows.append({
            "job_description": job_text,
            "resume": resume_text,
            "label": int(label),
            "source_file": fname,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No usable rows parsed from the dataset. Check labeling rules in src/data.py.")

    print(f"[HF] Final dataframe shape: {df.shape}")
    return df


def detect_schema(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    resume_col = _find_col(df, TEXT_COL_CANDIDATES["resume"])
    job_col = _find_col(df, TEXT_COL_CANDIDATES["job"])
    label_col = _find_col(df, LABEL_COL_CANDIDATES)
    score_col = _find_col(df, SCORE_COL_CANDIDATES)

    return {
        "resume_col": resume_col,
        "job_col": job_col,
        "label_col": label_col,
        "score_col": score_col,
    }


def build_xy(
    df: pd.DataFrame,
    score_threshold: float = CONFIG.default_score_threshold
) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
    schema = detect_schema(df)
    resume_col = schema["resume_col"]
    job_col = schema["job_col"]
    label_col = schema["label_col"]
    score_col = schema["score_col"]

    if not resume_col or not job_col:
        raise ValueError(
            f"Could not auto-detect resume/job columns. Found schema={schema}. "
            f"Adjust TEXT_COL_CANDIDATES in src/data.py."
        )

    job_texts = df[job_col].map(safe_str)
    resume_texts = df[resume_col].map(safe_str)

    if label_col:
        y = df[label_col].astype(int)
        label_strategy = f"Using label column: {label_col}"
    elif score_col:
        scores = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0)
        y = (scores >= score_threshold).astype(int)
        label_strategy = f"Using score column: {score_col} with threshold={score_threshold}"
    else:
        raise ValueError(f"No label or score column found. Schema detected: {schema}")

    X = pd.Series([build_pair_text(j, r) for j, r in zip(job_texts, resume_texts)])

    info = {
        "schema": schema,
        "label_strategy": label_strategy,
        "n_samples": len(df),
    }

    return X, y, info
