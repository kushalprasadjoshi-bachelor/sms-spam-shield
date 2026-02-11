from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class ModelType(str, Enum):
    LOGISTIC_REGRESSION = "lr"
    NAIVE_BAYES = "nb"
    SVM = "svm"
    LSTM = "lstm"


class PredictionRequest(BaseModel):
    sms: str = Field(..., min_length=1, max_length=1000)
    models: List[ModelType] = Field(default_factory=lambda: [ModelType.LOGISTIC_REGRESSION])
    include_explanation: bool = True
    
    class Config:
        json_schema_extra = {
            "example": {
                "sms": "Congratulations! You won a free iPhone. Click here to claim.",
                "models": ["lr", "nb"],
                "include_explanation": True
            }
        }


class Explanation(BaseModel):
    important_tokens: List[str] = []
    feature_importance: Dict[str, float] = {}
    confidence: float = 0.0


class ModelPrediction(BaseModel):
    model: ModelType
    prediction: str
    confidence: float
    explanation: Optional[Explanation] = None


class PredictionResponse(BaseModel):
    sms: str
    ensemble_prediction: Optional[str] = None
    ensemble_confidence: Optional[float] = None
    individual_predictions: List[ModelPrediction] = []
    processing_time_ms: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "sms": "Congratulations! You won a free iPhone. Click here to claim.",
                "ensemble_prediction": "phishing",
                "ensemble_confidence": 0.92,
                "individual_predictions": [
                    {
                        "model": "lr",
                        "prediction": "phishing",
                        "confidence": 0.94,
                        "explanation": {
                            "important_tokens": ["won", "free", "click", "claim"],
                            "confidence": 0.94
                        }
                    }
                ],
                "processing_time_ms": 45.2
            }
        }