from typing import Optional
from pathlib import Path
import io

import pdfplumber
import docx

from .utils import normalize_whitespace


def read_txt(file_bytes: bytes) -> str:
    """
    Reads text from a .txt file.
    """
    try:
        return normalize_whitespace(file_bytes.decode("utf-8", errors="ignore"))
    except Exception:
        return ""


def read_docx(file_bytes: bytes) -> str:
    """
    Extracts text from .docx resumes.
    """
    try:
        f = io.BytesIO(file_bytes)
        document = docx.Document(f)
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        return normalize_whitespace(text)
    except Exception:
        return ""


def read_pdf(file_bytes: bytes) -> str:
    """
    Extracts text from PDF resumes using pdfplumber.
    """
    try:
        f = io.BytesIO(file_bytes)
        text_parts = []
        with pdfplumber.open(f) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text() or ""
                if extracted.strip():
                    text_parts.append(extracted)
        return normalize_whitespace("\n".join(text_parts))
    except Exception:
        return ""


def extract_text(filename: str, file_bytes: bytes) -> str:
    """
    Detect file extension and extract text accordingly.
    """

    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        return read_pdf(file_bytes)

    if ext == ".docx":
        return read_docx(file_bytes)

    # treat .txt and unknown formats as text
    return read_txt(file_bytes)