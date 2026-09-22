"""
End-to-end training script for the Support Ticket Classification pipeline.

Usage:
    python -m src.run_training

This script:
1. Loads the raw dataset CSV
2. Cleans text and derives priority labels
3. Extracts TF-IDF features
4. Trains category and priority classifiers (comparing 4 algorithms each)
5. Evaluates on held-out test set
6. Saves model artifacts to models/ and evaluation reports to reports/
"""

import sys
import logging
import time

# Configure logging before any other imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("training")

# Suppress verbose library loggers
logging.getLogger("sklearn").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def main():
    """Run the full training pipeline."""
    start_time = time.time()

    logger.info("=" * 70)
    logger.info("SUPPORT TICKET CLASSIFIER — TRAINING PIPELINE")
    logger.info("=" * 70)

    # ---------------------------------------------------------------
    # Step 1: Load and prepare data
    # ---------------------------------------------------------------
    logger.info("\n[Step 1/5] Loading and preparing dataset...")
    from src.preprocessing import prepare_dataset
    X_train, X_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test, df = prepare_dataset()

    logger.info("Dataset prepared: %d train, %d test", len(X_train), len(X_test))

    # ---------------------------------------------------------------
    # Step 2: Extract features
    # ---------------------------------------------------------------
    logger.info("\n[Step 2/5] Extracting TF-IDF features...")
    from src.features import build_tfidf_vectorizer, transform_features

    vectorizer, X_train_feat = build_tfidf_vectorizer(X_train)
    X_test_feat = transform_features(X_test, vectorizer)

    logger.info("Feature extraction complete: %d features", X_train_feat.shape[1])

    # ---------------------------------------------------------------
    # Step 3: Train classifiers
    # ---------------------------------------------------------------
    logger.info("\n[Step 3/5] Training classifiers...")
    from src.train import train_classifier

    # Category classifier
    cat_model, cat_cv_results = train_classifier(
        X_train_feat, y_cat_train, task_name="Category"
    )

    # Priority classifier
    pri_model, pri_cv_results = train_classifier(
        X_train_feat, y_pri_train, task_name="Priority"
    )

    # ---------------------------------------------------------------
    # Step 4: Evaluate on test set
    # ---------------------------------------------------------------
    logger.info("\n[Step 4/5] Evaluating on test set...")
    from src.evaluate import evaluate_model, generate_summary_report
    from src.config import CATEGORY_LABELS, PRIORITY_LABELS

    cat_metrics = evaluate_model(
        cat_model, X_test_feat, y_cat_test, CATEGORY_LABELS, "Category"
    )

    pri_metrics = evaluate_model(
        pri_model, X_test_feat, y_pri_test, PRIORITY_LABELS, "Priority"
    )

    # Generate combined report
    generate_summary_report(cat_metrics, pri_metrics, cat_cv_results, pri_cv_results)

    # ---------------------------------------------------------------
    # Step 5: Save artifacts
    # ---------------------------------------------------------------
    logger.info("\n[Step 5/5] Saving model artifacts...")
    from src.train import save_artifacts

    save_artifacts(
        vectorizer=vectorizer,
        category_model=cat_model,
        priority_model=pri_model,
        category_cv_results=cat_cv_results,
        priority_cv_results=pri_cv_results,
        category_eval=cat_metrics,
        priority_eval=pri_metrics,
    )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE — %.1f seconds", elapsed)
    logger.info("=" * 70)
    logger.info("Category — Accuracy: %.4f | F1 (macro): %.4f", cat_metrics["accuracy"], cat_metrics["f1_macro"])
    logger.info("Priority — Accuracy: %.4f | F1 (macro): %.4f", pri_metrics["accuracy"], pri_metrics["f1_macro"])
    logger.info("Models saved to:  models/")
    logger.info("Reports saved to: reports/")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
