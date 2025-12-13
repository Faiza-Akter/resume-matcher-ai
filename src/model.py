import argparse
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .config import CONFIG
from .utils import ensure_dir, save_json
from .data import load_hf_dataframe, build_xy
from .evaluate import compute_metrics, metrics_to_dict


def train_and_save(score_threshold: float = CONFIG.default_score_threshold) -> dict:
    """
    Train the AI Resume Matcher using:
    - TF-IDF vectorizer
    - Logistic Regression classifier

    Saves:
    - vectorizer.joblib
    - classifier.joblib
    - label_info.json (dataset schema + metrics)
    """
    print("\nLoading dataset from HuggingFace...")
    df = load_hf_dataframe(CONFIG.hf_dataset_name)

    print("Building training features...")
    X, y, info = build_xy(df, score_threshold=score_threshold)

    # Safety: ensure there are 2 classes for training
    if y.nunique() < 2:
        raise ValueError(
            "Labeling produced only one class (all 0s or all 1s). "
            "Check filename labeling rules in src/data.py."
        )

    print("Splitting training/testing set...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG.test_size,
        random_state=CONFIG.random_state,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=CONFIG.max_features,
        ngram_range=CONFIG.ngram_range,
        stop_words="english",
    )

    print("Vectorizing text...")
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=2000)
    print("Training classifier...")
    clf.fit(Xtr, y_train)

    print("Computing accuracy + metrics...")
    y_pred = clf.predict(Xte)
    m = compute_metrics(y_test, y_pred)

    ensure_dir(CONFIG.model_dir)
    joblib.dump(vectorizer, CONFIG.vectorizer_path)
    joblib.dump(clf, CONFIG.classifier_path)

    label_info = {
        "dataset": CONFIG.hf_dataset_name,
        "score_threshold": score_threshold,
        "train_test_split": {
            "test_size": CONFIG.test_size,
            "random_state": CONFIG.random_state,
        },
        "data_info": info,
        "metrics": metrics_to_dict(m),
    }

    save_json(CONFIG.label_info_path, label_info)

    print("\nTraining complete! Model saved in 'models/' folder.")
    print("Model Performance:")
    print(label_info["metrics"])

    return label_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train and save model artifacts.")
    parser.add_argument("--threshold", type=float, default=CONFIG.default_score_threshold,
                        help="Score threshold for labeling.")
    args = parser.parse_args()

    if args.train:
        train_and_save(score_threshold=args.threshold)
    else:
        print("Nothing to do. Use:\n  py -m src.model --train\nTip: use unbuffered output:\n  py -u -m src.model --train")


if __name__ == "__main__":
    main()
