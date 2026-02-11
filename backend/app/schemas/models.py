from pydantic import BaseModel
from typing import Dict, Any
from datetime import datetime


class ModelInfo(BaseModel):
    name: str
    type: str
    status: str  # "loaded", "not_loaded", "training"
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_date: datetime
    parameters: Dict[str, Any]
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "logistic_regression",
                "type": "Logistic Regression",
                "status": "loaded",
                "accuracy": 0.92,
                "precision": 0.91,
                "recall": 0.93,
                "f1_score": 0.92,
                "training_date": "2024-01-15T10:30:00",
                "parameters": {
                    "C": 1.0,
                    "max_iter": 1000,
                    "solver": "lbfgs"
                }
            }
        }