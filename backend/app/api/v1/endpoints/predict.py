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
            model_types=request.models
        )
        
        # Format individual predictions
        individual_predictions = []
        for pred in prediction_result["individual_predictions"]:
            model_prediction = ModelPrediction(
                model=ModelType(pred["model"]),
                prediction=pred["prediction"],
                confidence=pred["confidence"]
            )
            
            # Add explanation if requested
            if request.include_explanation:
                # For now, simple token-based explanation
                # This will be enhanced with LIME/SHAP later
                important_tokens = extract_important_tokens(request.sms, pred["model"])
                explanation = Explanation(
                    important_tokens=important_tokens,
                    confidence=pred["confidence"]
                )
                model_prediction.explanation = explanation
            
            individual_predictions.append(model_prediction)
        
        processing_time = (time.time() - start_time) * 1000
        
        response = PredictionResponse(
            sms=request.sms,
            ensemble_prediction=prediction_result["ensemble_prediction"],
            ensemble_confidence=prediction_result["ensemble_confidence"],
            individual_predictions=individual_predictions,
            processing_time_ms=processing_time
        )
        
        logger.info(f"Prediction completed in {processing_time:.2f}ms")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


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