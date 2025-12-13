import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from .utils import normalize_whitespace

_NLTK_READY = False


def ensure_nltk():
    global _NLTK_READY
    if _NLTK_READY:
        return

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

    _NLTK_READY = True


def clean_text(text: str) -> str:
    ensure_nltk()

    if text is None:
        text = ""
    else:
        text = str(text)

    text = text.lower()
    text = normalize_whitespace(text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = word_tokenize(text)

    sw = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in sw and t.isalpha()]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def build_pair_text(job_text: str, resume_text: str) -> str:
    job_text = clean_text(job_text)
    resume_text = clean_text(resume_text)
    return f"JOB: {job_text} RESUME: {resume_text}"
