from fastapi import APIRouter, HTTPException
import time
from typing import List

from backend.app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    ModelPrediction,
    Explanation,
    ModelType
)
from backend.app.services.model_manager import model_manager
from backend.app.core.logger import logger
from backend.app.services.monitoring_service import monitoring_service

router = APIRouter()


@router.get("/models")
async def get_models_info():
    """Get information about all available models."""
    try:
        models_info = model_manager.get_all_models_info()
        return {
            "models": models_info,
            "loaded_count": sum(1 for m in models_info.values() if m["status"] == "loaded"),
            "total_count": len(models_info)
        }
    except Exception as e:
        logger.error(f"Error getting models info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/predict", response_model=PredictionResponse)
async def predict_sms(request: PredictionRequest):
    start_time = time.time()

    try:
        logger.info(f"Prediction request for SMS: {request.sms[:100]}...")
        logger.info(f"Selected models: {[m.value for m in request.models]}")

        prediction_result = model_manager.predict(
            text=request.sms,
            model_types=request.models,
            include_explanation=request.include_explanation
        )

        individual_predictions = []
        for pred in prediction_result["individual_predictions"]:
            explanation_obj = None
            if request.include_explanation and "explanation" in pred:
                exp = pred["explanation"]
                explanation_obj = Explanation(
                    important_tokens=[t["word"] for t in exp.get("important_tokens", [])],
                    confidence=pred["confidence"],
                    method=exp.get("method", "unknown")
                )
            model_pred = ModelPrediction(
                model=ModelType(pred["model"]),
                prediction=pred["prediction"],
                confidence=pred["confidence"],
                explanation=explanation_obj
            )
            individual_predictions.append(model_pred)

        processing_time = (time.time() - start_time) * 1000

        response = PredictionResponse(
            sms=request.sms,
            ensemble_prediction=prediction_result["ensemble_prediction"],
            ensemble_confidence=prediction_result["ensemble_confidence"],
            individual_predictions=individual_predictions,
            processing_time_ms=processing_time
        )

        # Record for monitoring
        monitoring_service.record_prediction(
            model='ensemble',
            category=response.ensemble_prediction or "unknown",
            confidence=response.ensemble_confidence or 0.0
        )
        for ind in individual_predictions:
            monitoring_service.record_prediction(
                model=ind.model.value,
                category=ind.prediction,
                confidence=ind.confidence
            )

        logger.info(f"Prediction completed in {processing_time:.2f}ms")
        return response

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/ensemble", response_model=PredictionResponse)
async def ensemble_predict(
    request: PredictionRequest,
    method: str = "weighted_voting"
):
    start_time = time.time()
    try:
        result = model_manager.ensemble_predict(
            text=request.sms,
            model_types=request.models,
            method=method,
            include_explanation=request.include_explanation
        )

        individual = []
        for ind in result["individual_predictions"]:
            model_pred = ModelPrediction(
                model=ModelType(ind["model"]),
                prediction=ind["prediction"],
                confidence=ind["confidence"],
                explanation=None
            )
            individual.append(model_pred)

        processing_time = (time.time() - start_time) * 1000
        response = PredictionResponse(
            sms=request.sms,
            ensemble_prediction=result["ensemble_prediction"],
            ensemble_confidence=result["ensemble_confidence"],
            individual_predictions=individual,
            processing_time_ms=processing_time
        )
        return response
    except Exception as e:
        logger.error(f"Ensemble prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))