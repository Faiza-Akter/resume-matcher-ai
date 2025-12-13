from typing import Tuple, Optional, Dict, Any
import pandas as pd
from datasets import load_dataset

from .config import CONFIG
from .utils import safe_str
from .preprocess import build_pair_text

# Column name candidates for auto-detection
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


# Auto-detect columns based on likely names
def _find_col(df: pd.DataFrame, candidates) -> Optional[str]:
    cols = set(df.columns)

    # Case-sensitive match
    for c in candidates:
        if c in cols:
            return c

    # Case-insensitive match
    lower_map = {x.lower(): x for x in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    return None


# Load HuggingFace dataset into DataFrame
def load_hf_dataframe(dataset_name: str = CONFIG.hf_dataset_name) -> pd.DataFrame:
    """
    Loads a dataset from HuggingFace Hub.
    Will automatically use 'train' split if present.
    """
    ds = load_dataset(dataset_name)

    if "train" in ds:
        df = ds["train"].to_pandas()
    else:
        split = list(ds.keys())[0]
        df = ds[split].to_pandas()

    return df


# Detect schema
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


# Build (X,y) training data
def build_xy(
    df: pd.DataFrame,
    score_threshold: float = CONFIG.default_score_threshold
) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
    """
    Returns:
        X_text: Combined (job + resume) cleaned text
        y: Labels (0/1)
        info: metadata about dataset and column mappings
    """

    schema = detect_schema(df)
    resume_col = schema["resume_col"]
    job_col = schema["job_col"]
    label_col = schema["label_col"]
    score_col = schema["score_col"]

    if not resume_col or not job_col:
        raise ValueError(
            f"Could not auto-detect resume/job columns. Found schema={schema}.\n"
            f"Please adjust TEXT_COL_CANDIDATES in src/data.py."
        )

    job_texts = df[job_col].map(safe_str)
    resume_texts = df[resume_col].map(safe_str)

    # If label column exists, use it
    if label_col:
        y = df[label_col].astype(int)
        label_strategy = f"Using label column: {label_col}"

    # If only score available, convert to binary label
    elif score_col:
        scores = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0)
        y = (scores >= score_threshold).astype(int)
        label_strategy = (
            f"Using score column: {score_col} with threshold={score_threshold}"
        )

    else:
        raise ValueError(
            f"No label or score column found. Schema detected: {schema}"
        )

    # Build text pairs
    X = pd.Series(
        [build_pair_text(j, r) for j, r in zip(job_texts, resume_texts)]
    )

    info = {
        "schema": schema,
        "label_strategy": label_strategy,
        "n_samples": len(df)
    }

    return X, y, info
