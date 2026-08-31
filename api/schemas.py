"""
Pydantic schemas for FastAPI request/response validation.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class PredictionInput(BaseModel):
    """Input schema for the /predict endpoint."""
    features: List[float] = Field(
        ...,
        description="List of feature values in the correct order",
        example=[30.0, 2.0, 1.0, 3.0, 0.0, 1.0, 0.0, 2.0, 5.0, 1.0, 1.0, 999.0, 0.0, 1.0, 0.0, 93.0, -40.0, 4.8, 5000.0]
    )

class PredictionOutput(BaseModel):
    """Output schema for the /predict endpoint."""
    prediction: int = Field(..., description="Predicted class (0 or 1)")
    probability: float = Field(..., description="Probability of class 1")
    model_used: str = Field(..., description="Name of the model used")
    confidence: float = Field(..., description="Confidence percentage")

class PredictionBatchInput(BaseModel):
    """Input schema for batch predictions."""
    features: List[List[float]] = Field(..., description="List of feature vectors")

class PredictionBatchOutput(BaseModel):
    """Output schema for batch predictions."""
    predictions: List[int]
    probabilities: List[float]
    model_used: str

class HealthCheck(BaseModel):
    """Health check response."""
    status: str
    models_loaded: List[str]
    features_count: int

class FeatureInfo(BaseModel):
    """Feature information response."""
    features: List[str]
    count: int
    description: str
