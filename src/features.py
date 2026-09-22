"""
Feature extraction for the Support Ticket Classification pipeline.

Provides TF-IDF (primary) and Bag-of-Words (alternative) vectorizers
for converting cleaned ticket text into numerical feature matrices.
"""

import logging
from typing import Tuple

import numpy as np
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from src.config import TFIDF_PARAMS, BOW_PARAMS

logger = logging.getLogger(__name__)


def build_tfidf_vectorizer(
    texts: np.ndarray,
) -> Tuple[TfidfVectorizer, spmatrix]:
    """
    Fit a TF-IDF vectorizer on the training texts.

    TF-IDF (Term Frequency–Inverse Document Frequency) captures how important
    a word is to a document relative to the entire corpus. Words that appear
    frequently in one document but rarely across all documents get higher weights.

    Configuration is controlled by TFIDF_PARAMS in config.py:
    - max_features: vocabulary size cap (10,000)
    - ngram_range: (1,2) captures single words AND two-word phrases
    - min_df/max_df: filters extremely rare and extremely common terms
    - sublinear_tf: dampens the effect of very high term frequencies

    Args:
        texts: Array of cleaned text strings (training set only).

    Returns:
        Tuple of (fitted vectorizer, feature matrix).
    """
    logger.info(
        "Building TF-IDF vectorizer (max_features=%d, ngrams=%s)...",
        TFIDF_PARAMS["max_features"],
        TFIDF_PARAMS["ngram_range"],
    )

    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X = vectorizer.fit_transform(texts)

    logger.info(
        "TF-IDF matrix shape: %s (%.1f MB sparse)",
        X.shape,
        X.data.nbytes / 1_000_000,
    )

    return vectorizer, X


def build_bow_vectorizer(
    texts: np.ndarray,
) -> Tuple[CountVectorizer, spmatrix]:
    """
    Fit a Bag-of-Words vectorizer on the training texts.

    BoW counts how many times each word appears in a document, without
    considering importance across the corpus. Simpler than TF-IDF but
    can still be effective for text classification.

    This is provided as a documented alternative for comparison.
    In practice, TF-IDF generally outperforms BoW for classification tasks
    because it downweights common words that carry little discriminative value.

    Args:
        texts: Array of cleaned text strings (training set only).

    Returns:
        Tuple of (fitted vectorizer, feature matrix).
    """
    logger.info(
        "Building BoW vectorizer (max_features=%d, ngrams=%s)...",
        BOW_PARAMS["max_features"],
        BOW_PARAMS["ngram_range"],
    )

    vectorizer = CountVectorizer(**BOW_PARAMS)
    X = vectorizer.fit_transform(texts)

    logger.info("BoW matrix shape: %s", X.shape)

    return vectorizer, X


def transform_features(texts, vectorizer) -> spmatrix:
    """
    Transform new texts using a previously fitted vectorizer.

    Used for:
    - Transforming test set features during evaluation
    - Transforming new ticket text during prediction (backend)

    Args:
        texts: Array of cleaned text strings, or a single string.
        vectorizer: A fitted TfidfVectorizer or CountVectorizer.

    Returns:
        Sparse feature matrix.
    """
    if isinstance(texts, str):
        texts = [texts]
    return vectorizer.transform(texts)
