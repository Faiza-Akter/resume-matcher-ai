from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    model_dir: Path = Path("models")

    vectorizer_path: Path = Path("models/vectorizer.joblib")
    classifier_path: Path = Path("models/classifier.joblib")
    label_info_path: Path = Path("models/label_info.json")

    hf_dataset_name: str = "netsol/resume-score-details"
    default_score_threshold: float = 50.0

    test_size: float = 0.2
    random_state: int = 42

    max_features: int = 40000
    ngram_range: tuple = (1, 2)


CONFIG = AppConfig()
