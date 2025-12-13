from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    # Directory to store trained ML artifacts
    model_dir: Path = Path("models")

    # Paths for saving vectorizer & classifier
    vectorizer_path: Path = Path("models/vectorizer.joblib")
    classifier_path: Path = Path("models/classifier.joblib")
    label_info_path: Path = Path("models/label_info.json")

    # HuggingFace dataset for resume matching
    hf_dataset_name: str = "netsol/resume-score-details"

    # Convert numeric score → label using a threshold (if dataset has scores)
    default_score_threshold: float = 50.0

    # Train/test split ratio
    test_size: float = 0.2
    random_state: int = 42

    # TF-IDF settings
    max_features: int = 40000
    ngram_range: tuple = (1, 2)


CONFIG = AppConfig()
