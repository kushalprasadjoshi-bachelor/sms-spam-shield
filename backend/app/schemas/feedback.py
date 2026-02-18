from datetime import datetime
from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class FeedbackRequest(BaseModel):
    prediction_id: str = Field(..., min_length=1, max_length=120)
    corrected_label: str = Field(..., min_length=1, max_length=64)
    predicted_label: Optional[str] = Field(default=None, max_length=64)
    sms: Optional[str] = Field(default="", max_length=1000)
    selected_models: List[str] = Field(default_factory=list)


class FeedbackRecord(BaseModel):
    prediction_id: str
    corrected_label: str
    predicted_label: Optional[str] = None
    sms: str = ""
    selected_models: List[str] = Field(default_factory=list)
    is_correct: Optional[bool] = None
    created_at: datetime
    updated_at: datetime


class FeedbackSubmitResponse(BaseModel):
    success: bool = True
    message: str
    feedback: FeedbackRecord


class FeedbackStatsResponse(BaseModel):
    total_feedback: int = 0
    total_corrections: int = 0
    total_confirmed_correct: int = 0
    corrections_by_model: Dict[str, int] = Field(default_factory=dict)
    corrections_by_label: Dict[str, int] = Field(default_factory=dict)
