"""
Configuration for the Support Ticket Classification & Prioritization pipeline.

This is the single source of truth for:
- Dataset column mapping (swap datasets by changing DATASET_CONFIG)
- Priority derivation rules (when no priority label exists)
- Model training hyperparameters
- File paths for data, models, and reports

To use a different dataset:
1. Place your CSV in DATA_RAW_DIR
2. Update DATASET_CONFIG with your column names
3. If your dataset has a priority column, set 'priority_column' and 'derive_priority' = False
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure output directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
DATASET_CONFIG = {
    # Filename inside DATA_RAW_DIR
    "filename": "all_tickets_processed_improved_v3.csv",

    # Column mapping — map your dataset's column names to the pipeline's fields
    "text_column": "Document",           # Column containing ticket text
    "category_column": "Topic_group",    # Column containing category labels
    "priority_column": None,             # Column containing priority labels (None if absent)

    # If priority_column is None, derive priority using rules below
    "derive_priority": True,

    # CSV reading options
    "encoding": "utf-8",
    "separator": ",",
}

# ---------------------------------------------------------------------------
# Category labels (auto-detected from data if empty)
# ---------------------------------------------------------------------------
CATEGORY_LABELS = [
    "Access",
    "Administrative rights",
    "HR Support",
    "Hardware",
    "Internal Project",
    "Miscellaneous",
    "Purchase",
    "Storage",
]

# ---------------------------------------------------------------------------
# Priority labels and derivation rules
# ---------------------------------------------------------------------------
PRIORITY_LABELS = ["Low", "Medium", "High"]

# Base priority assigned to each category (before keyword/complexity modifiers)
# Scale: 0 = Low, 1 = Medium, 2 = High
CATEGORY_BASE_PRIORITY = {
    "Access": 2,                  # Account access issues are typically urgent
    "Administrative rights": 2,   # Permission problems block work
    "Hardware": 1,                # Hardware issues vary in urgency
    "Storage": 1,                 # Storage issues can be blocking
    "HR Support": 0,              # Generally lower urgency
    "Purchase": 0,                # Procurement is planned, rarely urgent
    "Internal Project": 0,        # Project-related, usually not urgent
    "Miscellaneous": 0,           # Default low
}

# Keywords that signal urgency — presence bumps priority up by 1 level
URGENCY_KEYWORDS = [
    "urgent", "asap", "critical", "emergency", "immediately", "broken",
    "down", "blocked", "crashed", "failure", "outage", "expired",
    "expire", "locked", "unauthorized", "unable", "cannot access",
    "password reset", "security", "breach", "lost data", "deadline",
    "production", "server down", "not working", "help asap",
]

# Text length threshold (word count) — tickets above this are bumped up by 1
# (very long tickets often indicate complex/escalated issues)
COMPLEXITY_WORD_THRESHOLD = 120  # ~90th percentile of dataset

# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------
PREPROCESSING = {
    "lowercase": True,
    "remove_punctuation": True,
    "remove_numbers": False,       # Numbers can carry meaning (error codes, etc.)
    "remove_stopwords": True,
    "lemmatize": True,
    "min_token_length": 2,         # Drop single-character tokens
}

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
TFIDF_PARAMS = {
    "max_features": 10000,
    "ngram_range": (1, 2),        # Unigrams + bigrams
    "min_df": 2,                  # Ignore terms appearing in < 2 documents
    "max_df": 0.95,               # Ignore terms appearing in > 95% of documents
    "sublinear_tf": True,         # Apply sublinear TF scaling (1 + log(tf))
}

BOW_PARAMS = {
    "max_features": 10000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "binary": False,
}

# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
CROSS_VALIDATION_FOLDS = 5
RANDOM_STATE = 42
TEST_SIZE = 0.2  # 80/20 train/test split

# Model version tracking
MODEL_VERSION = "1.0.0"
