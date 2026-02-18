from fastapi import APIRouter, HTTPException

from ....core.logger import logger
from ....schemas.feedback import (
    FeedbackRequest,
    FeedbackStatsResponse,
    FeedbackSubmitResponse
)
from ....services.feedback_service import feedback_service

router = APIRouter()


@router.post("/feedback", response_model=FeedbackSubmitResponse)
async def submit_feedback(request: FeedbackRequest):
    """Store or update user feedback for a prediction."""
    try:
        feedback = feedback_service.upsert_feedback(request)
        return FeedbackSubmitResponse(
            success=True,
            message="Feedback saved successfully",
            feedback=feedback
        )
    except Exception as exc:
        logger.error(f"Failed to store feedback: {exc}")
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {exc}")


@router.get("/feedback/stats", response_model=FeedbackStatsResponse)
async def get_feedback_stats():
    """Return aggregated feedback statistics."""
    try:
        stats = feedback_service.get_stats()
        return FeedbackStatsResponse(**stats)
    except Exception as exc:
        logger.error(f"Failed to fetch feedback stats: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch feedback stats: {exc}")
