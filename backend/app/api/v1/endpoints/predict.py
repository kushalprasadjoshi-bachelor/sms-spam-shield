from fastapi import APIRouter, HTTPException
import time
from typing import Any, Dict, List, Tuple

from ....schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    ModelPrediction,
    Explanation,
    ModelType
)
from ....services.model_manager import model_manager
from ....core.logger import logger
from ....services.monitoring_service import monitoring_service

router = APIRouter()


def _normalize_explanation_tokens(exp: Dict[str, Any]) -> Tuple[List[str], Dict[str, float]]:
    """Normalize explanation payloads that may contain token strings or token objects."""
    token_items = exp.get("important_tokens", [])
    important_tokens: List[str] = []
    feature_importance: Dict[str, float] = {}

    if isinstance(token_items, list):
        for item in token_items:
            if isinstance(item, dict):
                word = str(item.get("word", "")).strip()
                if not word:
                    continue
                important_tokens.append(word)
                try:
                    feature_importance[word] = float(item.get("importance", 0.0))
                except (TypeError, ValueError):
                    continue
            elif isinstance(item, str):
                word = item.strip()
                if word:
                    important_tokens.append(word)

    raw_feature_importance = exp.get("feature_importance", {})
    if isinstance(raw_feature_importance, dict):
        for token, score in raw_feature_importance.items():
            word = str(token).strip()
            if not word:
                continue
            try:
                feature_importance[word] = float(score)
            except (TypeError, ValueError):
                continue

    if important_tokens:
        deduped_tokens: List[str] = []
        seen = set()
        for token in important_tokens:
            if token in seen:
                continue
            seen.add(token)
            deduped_tokens.append(token)
            if len(deduped_tokens) >= 12:
                break
        important_tokens = deduped_tokens
    elif feature_importance:
        important_tokens = list(feature_importance.keys())[:12]

    return important_tokens, feature_importance


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
        if not prediction_result["individual_predictions"]:
            raise HTTPException(
                status_code=503,
                detail="No selected models are loaded or predictions failed"
            )

        individual_predictions = []
        for pred in prediction_result["individual_predictions"]:
            explanation_obj = None
            exp = pred.get("explanation")
            if request.include_explanation and isinstance(exp, dict):
                important_tokens, feature_importance = _normalize_explanation_tokens(exp)
                explanation_obj = Explanation(
                    important_tokens=important_tokens,
                    feature_importance=feature_importance,
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
        if response.ensemble_prediction is not None:
            monitoring_service.record_prediction(
                model="ensemble",
                category=response.ensemble_prediction,
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

    except HTTPException:
        raise
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
            explanation_obj = None
            exp = ind.get("explanation")
            if request.include_explanation and isinstance(exp, dict):
                important_tokens, feature_importance = _normalize_explanation_tokens(exp)
                explanation_obj = Explanation(
                    important_tokens=important_tokens,
                    feature_importance=feature_importance,
                    confidence=ind["confidence"],
                    method=exp.get("method", "unknown")
                )
            model_pred = ModelPrediction(
                model=ModelType(ind["model"]),
                prediction=ind["prediction"],
                confidence=ind["confidence"],
                explanation=explanation_obj
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ensemble prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
