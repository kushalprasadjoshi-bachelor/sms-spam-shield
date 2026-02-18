from fastapi import APIRouter, HTTPException

from ....schemas.prediction import CompareRequest, ModelType
from ....services.model_manager import model_manager
from ....core.logger import logger

router = APIRouter()


@router.post("/compare")
async def compare_model_predictions(request: CompareRequest):
    """
    Compare predictions from all loaded models for the same SMS.

    - **sms**: SMS text to classify with all models
    """
    try:
        sms = request.sms
        logger.info(f"Model comparison request for SMS: {sms[:100]}...")

        selected_models = request.models if request.models else list(ModelType)
        if len(selected_models) < 2:
            raise HTTPException(
                status_code=400,
                detail="Please select at least two models for comparison"
            )

        results = model_manager.compare_models(sms, model_types=selected_models)
        if results["total_models"] < 2:
            raise HTTPException(
                status_code=400,
                detail="At least two selected models must be loaded to compare"
            )

        formatted_results = []
        for model_name, result in results["comparison"].items():
            formatted_results.append({
                "model": model_name,
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "status": result["status"],
                "error": result.get("error"),
                "params": result.get("params")
            })

        return {
            "sms": sms,
            "comparison": formatted_results,
            "summary": {
                "agreement": results["agreement"],
                "total_models": results["total_models"],
                "successful_models": results["successful_models"]
            }
        }

    except Exception as e:
        logger.error(f"Model comparison failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Model comparison failed: {str(e)}"
        )


@router.get("/ensemble/methods")
async def get_ensemble_methods():
    """
    Get available ensemble methods and descriptions.
    """
    return {
        "methods": [
            {
                "id": "weighted_voting",
                "name": "Weighted Voting",
                "description": "Each model vote is weighted by its confidence score",
                "formula": "score(pred) = sum(confidence_i for models predicting pred)"
            },
            {
                "id": "averaging",
                "name": "Probability Averaging",
                "description": "Average probability distributions from all models",
                "formula": "P_avg(c) = (1/n) * sum(P_i(c)) for i=1..n models"
            },
            {
                "id": "majority_voting",
                "name": "Majority Voting",
                "description": "Simple majority of model predictions",
                "formula": "pred = mode(predictions)"
            }
        ]
    }
