"""
Pydantic schemas for the Support Ticket Classification API.

Defines request/response models with validation for all endpoints.
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class TicketInput(BaseModel):
    """Single ticket prediction request."""
    text: str = Field(
        ...,
        min_length=3,
        max_length=50000,
        description="The support ticket text to classify.",
        examples=["My laptop screen is broken and I need a replacement urgently"],
    )

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Ticket text cannot be empty or whitespace-only.")
        return v.strip()


class BatchTicketInput(BaseModel):
    """Batch prediction request with a list of tickets."""
    tickets: List[TicketInput] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of ticket objects to classify (max 1000).",
    )


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class PredictionResult(BaseModel):
    """Prediction result for a single ticket."""
    text: str = Field(description="The original ticket text (truncated).")
    category: str = Field(description="Predicted category label.")
    category_confidence: float = Field(
        description="Confidence score for the predicted category (0–1).",
        ge=0.0, le=1.0,
    )
    priority: str = Field(description="Predicted priority level.")
    priority_confidence: float = Field(
        description="Confidence score for the predicted priority (0–1).",
        ge=0.0, le=1.0,
    )
    category_probabilities: Optional[Dict[str, float]] = Field(
        default=None,
        description="Probability distribution across all categories.",
    )
    priority_probabilities: Optional[Dict[str, float]] = Field(
        default=None,
        description="Probability distribution across all priority levels.",
    )


class BatchPredictionResult(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResult]
    total: int = Field(description="Total number of tickets processed.")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    model_loaded: bool = True


class ModelInfoResponse(BaseModel):
    """Model information response."""
    version: str
    trained_at: str
    category_model: str
    priority_model: str
    category_metrics: Optional[Dict] = None
    priority_metrics: Optional[Dict] = None
    cross_validation: Optional[Dict] = None


# ---------------------------------------------------------------------------
# Training schemas
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    """Training job request with configurable parameters."""
    algorithm: str = Field(
        default="LinearSVC",
        description="Algorithm to train: LinearSVC, LogisticRegression, RandomForest, MultinomialNB, or all.",
        examples=["LinearSVC", "all"],
    )
    feature_extraction: str = Field(
        default="tfidf",
        description="Feature extraction method: 'tfidf' or 'bow'.",
    )
    dataset: str = Field(
        default="all_tickets",
        description="Dataset identifier.",
    )
    cv_folds: int = Field(default=5, ge=2, le=10, description="Number of cross-validation folds.")
    test_split: float = Field(default=0.2, ge=0.1, le=0.5, description="Test set ratio.")
    max_features: int = Field(default=10000, ge=1000, le=50000, description="Max features for vectorizer.")

    @field_validator("algorithm")
    @classmethod
    def validate_algorithm(cls, v: str) -> str:
        allowed = {"LinearSVC", "LogisticRegression", "RandomForest", "MultinomialNB", "all"}
        if v not in allowed:
            raise ValueError(f"Algorithm must be one of {allowed}")
        return v

    @field_validator("feature_extraction")
    @classmethod
    def validate_feature(cls, v: str) -> str:
        if v not in {"tfidf", "bow"}:
            raise ValueError("feature_extraction must be 'tfidf' or 'bow'")
        return v


class TrainFoldResult(BaseModel):
    """Result for a single cross-validation fold."""
    fold: int
    accuracy: float
    f1: float
    loss: float


class TrainModelResult(BaseModel):
    """Cross-validation results for a single model."""
    name: str
    mean_accuracy: float
    std_accuracy: float
    mean_f1: float
    std_f1: float
    fold_results: List[TrainFoldResult] = []
    is_best: bool = False


class TrainStatusResponse(BaseModel):
    """Training job status response."""
    job_id: str
    status: str = Field(description="One of: queued, running, completed, failed")
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Progress 0.0-1.0")
    current_step: str = Field(default="", description="Human-readable current step")
    models_completed: List[TrainModelResult] = []
    category_results: Optional[Dict] = None
    priority_results: Optional[Dict] = None
    error: Optional[str] = None
    elapsed_seconds: Optional[float] = None
