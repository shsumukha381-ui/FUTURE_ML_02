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
