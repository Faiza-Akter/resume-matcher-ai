import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from .utils import normalize_whitespace

# Global flag to avoid downloading NLTK resources multiple times
_NLTK_READY = False

def ensure_nltk():
    """
    Ensure NLTK data is downloaded only once.
    """
    global _NLTK_READY
    if _NLTK_READY:
        return

    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

    _NLTK_READY = True


def clean_text(text: str) -> str:
    """
    Cleans text using your teacher’s NLP steps:
    - lowercase
    - remove punctuation
    - tokenize
    - remove stopwords
    - lemmatize tokens
    - join back into cleaned string
    """

    # Download resources (only first time)
    ensure_nltk()

    # Lowercase & remove noise
    text = text.lower()
    text = normalize_whitespace(text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    sw = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in sw and t.isalpha()]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def build_pair_text(job_text: str, resume_text: str) -> str:
    """
    Converts a (job description + resume) pair into a single text sample.

    This is important for ML training:
    Instead of trying to classify separately, we combine them like:

      "JOB: python developer requirements RESUME: experience in python django"

    """

    job_text = clean_text(job_text)
    resume_text = clean_text(resume_text)

    return f"JOB: {job_text} RESUME: {resume_text}"
