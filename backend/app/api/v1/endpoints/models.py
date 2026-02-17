from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from backend.app.services.model_manager import model_manager
from backend.app.schemas.prediction import ModelType
from backend.app.core.logger import logger

router = APIRouter()


@router.get("/versions/{model_name}")
async def list_model_versions(model_name: str):
    """List all versions of a specific model"""
    try:
        from ml.training.versioning import ModelVersionManager
        version_manager = ModelVersionManager(model_name)
        versions = version_manager.list_versions()
        return {"model": model_name, "versions": versions}
    except Exception as e:
        logger.error(f"Failed to list versions for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/versions/{model_name}/set-production")
async def set_production_version(model_name: str, version: str):
    """Set a specific version as production"""
    try:
        from ml.training.versioning import ModelVersionManager
        version_manager = ModelVersionManager(model_name)
        version_manager.set_production(version)
        return {"success": True, "model": model_name, "production_version": version}
    except Exception as e:
        logger.error(f"Failed to set production version for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feature-importance/{model_name}")
async def get_feature_importance(model_name: str, model_type: ModelType):
    """Get global feature importance for linear models"""
    model = model_manager.models.get(model_type)
    if not model or not model.loaded or not model.explainer:
        raise HTTPException(status_code=404, detail="Model not loaded or no explainer")
    
    importance = model.explainer.get_feature_importance()
    return {"model": model_name, "feature_importance": importance}