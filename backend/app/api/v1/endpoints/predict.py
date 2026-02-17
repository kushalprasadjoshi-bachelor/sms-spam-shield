#################### UNIFIED PREDICTION ENDPOINT #########################
from fastapi import APIRouter, HTTPException, Depends
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

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict_sms(request: PredictionRequest):
    """
    Predict SMS category using selected models.
    
    - **sms**: SMS text to classify
    - **models**: List of models to use (lr, nb, svm, lstm)
    - **include_explanation**: Whether to include explanation
    """
    start_time = time.time()

    try:
        logger.info(f"Prediction request for SMS: {request.sms[:100]}...")
        logger.info(f"Selected models: {[m.value for m in request.models]}")

        # Make predictions
        prediction_result = model_manager.predict(
            text=request.sms,
            model_types=request.models,
            include_explanation=request.include_explanation
        )

        # Format individual predictions
        individual_predictions = []
        for pred in prediction_result["individual_predictions"]:
            # Build explanation if present
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
            category=prediction_result["ensemble_prediction"] or "unknown",
            confidence=prediction_result["ensemble_confidence"] or 0.0,
            correct=None  # can be updated later if ground truth available
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


# Optional: endpoint for advanced ensemble
@router.post("/ensemble", response_model=PredictionResponse)
async def ensemble_predict(
    request: PredictionRequest,
    method: str = "weighted_voting"
):
    """Advanced ensemble prediction with method selection."""
    start_time = time.time()
    try:
        result = model_manager.ensemble_predict(
            text=request.sms,
            model_types=request.models,
            method=method,
            include_explanation=request.include_explanation
        )

        # Build individual predictions with explanations if available
        individual = []
        for ind in result["individual_predictions"]:
            # In ensemble_predict we may not have explanations per model yet
            # We could re-fetch them or trust that they were included in the call
            explanation_obj = None  # To be extended
            model_pred = ModelPrediction(
                model=ModelType(ind["model"]),
                prediction=ind["prediction"],
                confidence=ind["confidence"],
                explanation=explanation_obj
            )
            individual.append(model_pred)

        processing_time = (time.time() - start_time) * 1000
        return PredictionResponse(
            sms=request.sms,
            ensemble_prediction=result["ensemble_prediction"],
            ensemble_confidence=result["ensemble_confidence"],
            individual_predictions=individual,
            processing_time_ms=processing_time
        )
    except Exception as e:
        logger.error(f"Ensemble prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def extract_important_tokens(text: str, model_type: str) -> List[str]:
    """Extract important tokens from text (simplified for now)"""
    # This is a simplified version. We'll enhance this with actual model explanations later.
    from backend.app.services.preprocessing import preprocessor
    
    processed = preprocessor.preprocess(text)
    tokens = processed.split()
    
    # Simple heuristic: tokens that are not stopwords
    important_tokens = []
    for token in tokens:
        if len(token) > 3 and token not in preprocessor.stop_words:
            important_tokens.append(token)
    
    # Return top 5 important tokens
    return important_tokens[:5]


@router.get("/models")
async def get_models_info():
    """Get information about all available models"""
    try:
        models_info = model_manager.get_all_models_info()
        return {
            "models": models_info,
            "loaded_count": sum(1 for m in models_info.values() if m["status"] == "loaded"),
            "total_count": len(models_info)
        }
    except Exception as e:
        logger.error(f"Error getting models info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get models info: {str(e)}"
        )