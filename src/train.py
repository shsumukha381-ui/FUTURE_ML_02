"""
Model training for the Support Ticket Classification pipeline.

Trains and compares multiple classifiers using cross-validation,
selects the best, and serializes model artifacts for production use.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any

import numpy as np
import joblib
from scipy.sparse import spmatrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold

from src.config import (
    MODELS_DIR, CROSS_VALIDATION_FOLDS, RANDOM_STATE, MODEL_VERSION,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classifier definitions
# ---------------------------------------------------------------------------

def _get_classifiers() -> List[Tuple[str, Any]]:
    """
    Return the list of classifiers to evaluate.

    Each classifier is wrapped to ensure it supports `predict_proba`
    (needed for confidence scores in the API):
    - LogisticRegression and RandomForest support it natively.
    - LinearSVC and MultinomialNB are wrapped with CalibratedClassifierCV
      to provide probability estimates via Platt scaling.

    Returns:
        List of (name, classifier_instance) tuples.
    """
    return [
        (
            "LogisticRegression",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                solver="lbfgs",
                C=1.0,
            ),
        ),
        (
            "LinearSVC",
            CalibratedClassifierCV(
                LinearSVC(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    dual="auto",
                ),
                cv=3,  # Internal CV for calibration
            ),
        ),
        (
            "MultinomialNB",
            CalibratedClassifierCV(
                MultinomialNB(alpha=1.0),
                cv=3,
            ),
        ),
        (
            "RandomForest",
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                max_depth=None,
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Training with cross-validation
# ---------------------------------------------------------------------------

def train_classifier(
    X: spmatrix,
    y: np.ndarray,
    task_name: str,
) -> Tuple[Any, Dict[str, Dict[str, float]]]:
    """
    Train multiple classifiers, evaluate via cross-validation, and select the best.

    Uses stratified K-fold cross-validation with macro F1 as the selection metric.
    Macro F1 weights all classes equally, which is important for imbalanced datasets
    where we don't want the model to ignore minority classes.

    Args:
        X: Feature matrix (sparse, from TF-IDF).
        y: Target labels.
        task_name: Human-readable name for logging (e.g., 'Category', 'Priority').

    Returns:
        Tuple of (best_model, cv_results_dict).
        cv_results_dict maps model_name -> {mean_f1, std_f1, mean_accuracy, std_accuracy}.
    """
    logger.info("=" * 60)
    logger.info("Training classifiers for: %s", task_name)
    logger.info("=" * 60)
    logger.info("Training samples: %d, Features: %d", X.shape[0], X.shape[1])
    logger.info("Class distribution: %s", dict(zip(*np.unique(y, return_counts=True))))

    classifiers = _get_classifiers()
    cv = StratifiedKFold(n_splits=CROSS_VALIDATION_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    best_score = -1.0
    best_model = None
    best_name = ""

    for name, clf in classifiers:
        logger.info("\n  Training %s...", name)

        try:
            # Cross-validate on macro F1
            f1_scores = cross_val_score(
                clf, X, y, cv=cv, scoring="f1_macro", n_jobs=-1
            )
            acc_scores = cross_val_score(
                clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1
            )

            result = {
                "mean_f1": float(np.mean(f1_scores)),
                "std_f1": float(np.std(f1_scores)),
                "mean_accuracy": float(np.mean(acc_scores)),
                "std_accuracy": float(np.std(acc_scores)),
            }
            results[name] = result

            logger.info(
                "  %s — F1: %.4f (±%.4f)  Acc: %.4f (±%.4f)",
                name,
                result["mean_f1"], result["std_f1"],
                result["mean_accuracy"], result["std_accuracy"],
            )

            if result["mean_f1"] > best_score:
                best_score = result["mean_f1"]
                best_model = clf
                best_name = name

        except Exception as e:
            logger.error("  %s failed: %s", name, e)
            results[name] = {"error": str(e)}

    # Refit the best model on the full training set
    logger.info(
        "\n  >>> Best model for %s: %s (F1=%.4f) — refitting on full training set...",
        task_name, best_name, best_score,
    )
    best_model.fit(X, y)

    return best_model, results


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_artifacts(
    vectorizer,
    category_model,
    priority_model,
    category_cv_results: Dict,
    priority_cv_results: Dict,
    category_eval: Dict = None,
    priority_eval: Dict = None,
) -> None:
    """
    Serialize trained model artifacts and metadata to disk.

    Saves:
    - vectorizer.joblib: The fitted TF-IDF vectorizer
    - category_model.joblib: The best category classifier
    - priority_model.joblib: The best priority classifier
    - metadata.json: Version, training timestamp, CV results, evaluation metrics

    Args:
        vectorizer: Fitted TfidfVectorizer.
        category_model: Trained category classifier.
        priority_model: Trained priority classifier.
        category_cv_results: Cross-validation results for category models.
        priority_cv_results: Cross-validation results for priority models.
        category_eval: Optional test-set evaluation metrics for category model.
        priority_eval: Optional test-set evaluation metrics for priority model.
    """
    logger.info("Saving model artifacts to %s", MODELS_DIR)

    joblib.dump(vectorizer, MODELS_DIR / "vectorizer.joblib")
    joblib.dump(category_model, MODELS_DIR / "category_model.joblib")
    joblib.dump(priority_model, MODELS_DIR / "priority_model.joblib")

    metadata = {
        "version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "cross_validation": {
            "category": category_cv_results,
            "priority": priority_cv_results,
        },
    }
    if category_eval:
        metadata["evaluation"] = metadata.get("evaluation", {})
        metadata["evaluation"]["category"] = category_eval
    if priority_eval:
        metadata["evaluation"] = metadata.get("evaluation", {})
        metadata["evaluation"]["priority"] = priority_eval

    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Artifacts saved: vectorizer.joblib, category_model.joblib, priority_model.joblib, metadata.json")


def load_artifacts():
    """
    Load saved model artifacts from disk.

    Returns:
        Tuple of (vectorizer, category_model, priority_model, metadata_dict).

    Raises:
        FileNotFoundError: If model files are missing (need to run training first).
    """
    vectorizer_path = MODELS_DIR / "vectorizer.joblib"
    cat_path = MODELS_DIR / "category_model.joblib"
    pri_path = MODELS_DIR / "priority_model.joblib"
    meta_path = MODELS_DIR / "metadata.json"

    for path in [vectorizer_path, cat_path, pri_path, meta_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {path}. "
                f"Run 'python -m src.run_training' to train models first."
            )

    vectorizer = joblib.load(vectorizer_path)
    category_model = joblib.load(cat_path)
    priority_model = joblib.load(pri_path)

    with open(meta_path) as f:
        metadata = json.load(f)

    logger.info("Loaded model artifacts (version %s, trained %s)", metadata["version"], metadata["trained_at"])

    return vectorizer, category_model, priority_model, metadata
