import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import joblib

from backend.app.core.config import settings
from backend.app.core.logger import logger


class ModelVersionManager:
    """Manage model versions and metadata"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_dir = Path(settings.MODEL_REGISTRY_PATH) / model_name
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.versions_dir = self.model_dir / "versions"
        self.versions_dir.mkdir(exist_ok=True)
        
    def save_version(
        self,
        model,
        vectorizer,
        metrics: Dict[str, float],
        params: Dict[str, Any],
        version: Optional[str] = None
    ) -> str:
        """Save a new model version"""
        if version is None:
            # Generate version based on timestamp
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        version_dir = self.versions_dir / version
        version_dir.mkdir(exist_ok=True)
        
        # Save model and vectorizer
        joblib.dump(model, version_dir / "model.pkl")
        joblib.dump(vectorizer, version_dir / "vectorizer.pkl")
        
        # Save metadata
        metadata = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "model_name": self.model_name,
            "parameters": params,
            "metrics": metrics,
            "is_production": False
        }
        
        with open(version_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Also save a copy of metrics and metadata in main directory
        with open(self.model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        with open(self.model_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved {self.model_name} version {version}")
        return version
    
    def load_version(self, version: str):
        """Load a specific model version"""
        version_dir = self.versions_dir / version
        if not version_dir.exists():
            raise ValueError(f"Version {version} not found")
        
        model = joblib.load(version_dir / "model.pkl")
        vectorizer = joblib.load(version_dir / "vectorizer.pkl")
        
        with open(version_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        return model, vectorizer, metadata
    
    def list_versions(self) -> list:
        """List all available versions"""
        versions = []
        for version_dir in sorted(self.versions_dir.iterdir(), reverse=True):
            if version_dir.is_dir():
                metadata_path = version_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    versions.append({
                        "version": metadata["version"],
                        "created_at": metadata["created_at"],
                        "metrics": metadata["metrics"],
                        "is_production": metadata.get("is_production", False)
                    })
        return versions
    
    def set_production(self, version: str):
        """Set a version as production"""
        version_dir = self.versions_dir / version
        if not version_dir.exists():
            raise ValueError(f"Version {version} not found")
        
        # Update metadata for this version
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        metadata["is_production"] = True
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Copy to main directory as production model
        shutil.copy(version_dir / "model.pkl", self.model_dir / "model.pkl")
        shutil.copy(version_dir / "vectorizer.pkl", self.model_dir / "vectorizer.pkl")
        shutil.copy(version_dir / "metadata.json", self.model_dir / "metadata.json")
        
        logger.info(f"Set {self.model_name} version {version} as production")
    
    def get_production_version(self) -> Optional[dict]:
        """Get current production version metadata"""
        if (self.model_dir / "metadata.json").exists():
            with open(self.model_dir / "metadata.json", "r") as f:
                metadata = json.load(f)
            return metadata
        return None