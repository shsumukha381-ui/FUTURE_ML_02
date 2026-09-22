"""
Text preprocessing and data loading for the Support Ticket Classification pipeline.

Responsibilities:
- Load raw CSV with configurable column mapping
- Clean and normalize ticket text
- Derive priority labels when not present in the dataset
- Prepare train/test splits
"""

import re
import logging
from typing import Optional

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_RAW_DIR, DATASET_CONFIG, PREPROCESSING, RANDOM_STATE, TEST_SIZE,
    CATEGORY_BASE_PRIORITY, URGENCY_KEYWORDS, COMPLEXITY_WORD_THRESHOLD,
    PRIORITY_LABELS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLTK resource bootstrap
# ---------------------------------------------------------------------------
_NLTK_RESOURCES = ["stopwords", "wordnet", "omw-1.4"]

def _ensure_nltk_resources():
    """Download NLTK data files if not already present."""
    for resource in _NLTK_RESOURCES:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            logger.info("Downloading NLTK resource: %s", resource)
            nltk.download(resource, quiet=True)

_ensure_nltk_resources()

# Pre-load resources for performance
_STOP_WORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Clean and normalize a single ticket text string.

    Steps applied (controlled by config.PREPROCESSING):
    1. Lowercase
    2. Remove punctuation and special characters
    3. Remove extra whitespace
    4. Tokenize
    5. Remove stopwords
    6. Lemmatize
    7. Filter short tokens
    8. Rejoin

    Args:
        text: Raw ticket text string.

    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    cfg = PREPROCESSING

    # Lowercase
    if cfg["lowercase"]:
        text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", " ", text)

    # Remove punctuation and special characters (keep alphanumeric and spaces)
    if cfg["remove_punctuation"]:
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Remove numbers
    if cfg["remove_numbers"]:
        text = re.sub(r"\d+", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    tokens = text.split()

    # Remove stopwords
    if cfg["remove_stopwords"]:
        tokens = [t for t in tokens if t not in _STOP_WORDS]

    # Lemmatize
    if cfg["lemmatize"]:
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]

    # Filter short tokens
    min_len = cfg["min_token_length"]
    tokens = [t for t in tokens if len(t) >= min_len]

    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Priority derivation
# ---------------------------------------------------------------------------

def derive_priority(text: str, category: str) -> str:
    """
    Derive a priority label from ticket text and category using rule-based heuristics.

    The derivation combines three signals:
    1. Category base priority (some categories are inherently more urgent)
    2. Urgency keyword detection (words like 'urgent', 'blocked', 'critical')
    3. Text complexity (very long tickets suggest escalated/complex issues)

    The final score is clamped to [0, 2] and mapped to Low/Medium/High.

    This function is ONLY used when the dataset lacks a priority column.
    The logic is fully transparent and documented — see config.py for rules.

    Args:
        text: The raw (uncleaned) ticket text.
        category: The ticket's category label.

    Returns:
        Priority label: 'Low', 'Medium', or 'High'.
    """
    # Start with category base priority
    score = CATEGORY_BASE_PRIORITY.get(category, 0)

    # Check for urgency keywords in the original text (case-insensitive)
    text_lower = text.lower() if isinstance(text, str) else ""
    urgency_matches = sum(1 for kw in URGENCY_KEYWORDS if kw in text_lower)
    if urgency_matches >= 2:
        score += 2  # Multiple urgency signals = strong bump
    elif urgency_matches == 1:
        score += 1

    # Check text complexity (word count)
    word_count = len(text_lower.split()) if text_lower else 0
    if word_count > COMPLEXITY_WORD_THRESHOLD:
        score += 1

    # Clamp to valid range and map to label
    score = max(0, min(2, score))
    return PRIORITY_LABELS[score]  # 0=Low, 1=Medium, 2=High


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load the raw dataset CSV and standardize column names.

    Uses DATASET_CONFIG for column mapping so different datasets
    can be used without changing pipeline code.

    Args:
        filepath: Optional path to CSV. Defaults to DATA_RAW_DIR / config filename.

    Returns:
        DataFrame with columns: 'text', 'category', and optionally 'priority'.
    """
    cfg = DATASET_CONFIG

    if filepath is None:
        filepath = DATA_RAW_DIR / cfg["filename"]

    logger.info("Loading dataset from %s", filepath)
    df = pd.read_csv(
        filepath,
        encoding=cfg["encoding"],
        sep=cfg["separator"],
    )
    logger.info("Loaded %d rows, columns: %s", len(df), list(df.columns))

    # Map columns to standard names
    column_map = {cfg["text_column"]: "text"}
    if cfg["category_column"]:
        column_map[cfg["category_column"]] = "category"
    if cfg["priority_column"]:
        column_map[cfg["priority_column"]] = "priority"

    # Validate required columns exist
    for col in column_map:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in dataset. "
                f"Available columns: {list(df.columns)}. "
                f"Update DATASET_CONFIG in src/config.py."
            )

    df = df.rename(columns=column_map)

    # Drop rows with missing text or category
    initial_len = len(df)
    df = df.dropna(subset=["text", "category"])
    df = df[df["text"].str.strip().astype(bool)]
    dropped = initial_len - len(df)
    if dropped > 0:
        logger.warning("Dropped %d rows with missing text or category", dropped)

    return df


def prepare_dataset(df: Optional[pd.DataFrame] = None):
    """
    Full dataset preparation pipeline: load → clean → derive priority → split.

    Args:
        df: Optional pre-loaded DataFrame. If None, loads from configured path.

    Returns:
        Tuple of (X_train, X_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test, df_full)
        where X values are cleaned text strings.
    """
    if df is None:
        df = load_dataset()

    logger.info("Cleaning %d ticket texts...", len(df))
    df["text_clean"] = df["text"].apply(clean_text)

    # Drop rows where cleaning produced empty text
    empty_mask = df["text_clean"].str.strip() == ""
    if empty_mask.any():
        logger.warning("Dropping %d rows with empty text after cleaning", empty_mask.sum())
        df = df[~empty_mask].copy()

    # Derive priority if not present
    if DATASET_CONFIG["derive_priority"] or "priority" not in df.columns:
        logger.info("Deriving priority labels (no priority column in dataset)...")
        df["priority"] = df.apply(
            lambda row: derive_priority(row["text"], row["category"]), axis=1
        )
        pri_dist = df["priority"].value_counts()
        logger.info("Derived priority distribution:\n%s", pri_dist.to_string())

    # Log category distribution
    cat_dist = df["category"].value_counts()
    logger.info("Category distribution:\n%s", cat_dist.to_string())

    # Stratified train/test split (stratify on category since it's the primary task)
    X_train, X_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test = train_test_split(
        df["text_clean"].values,
        df["category"].values,
        df["priority"].values,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["category"].values,
    )

    logger.info(
        "Split: %d train / %d test (%.0f%% / %.0f%%)",
        len(X_train), len(X_test),
        100 * len(X_train) / len(df),
        100 * len(X_test) / len(df),
    )

    return X_train, X_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test, df
