import os
import json
import joblib
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from backend.app.core.config import settings
from backend.app.core.logger import logger
from backend.app.schemas.prediction import ModelType
from backend.app.services.preprocessing import preprocessor


class ModelMetadata:
    def __init__(self, model_type: ModelType):
        self.model_type = model_type
        self.model = None
        self.vectorizer = None
        self.metadata = {}
        self.metrics = {}
        self.loaded = False
    
    def load(self):
        """Load model from disk"""
        try:
            model_path = Path(settings.MODEL_REGISTRY_PATH) / self.model_type.value
            
            # Load model
            model_file = model_path / "model.pkl"
            if model_file.exists():
                self.model = joblib.load(model_file)
            
            # Load vectorizer
            vectorizer_file = model_path / "vectorizer.pkl"
            if vectorizer_file.exists():
                self.vectorizer = joblib.load(vectorizer_file)
            
            # Load metadata
            metadata_file = model_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            
            # Load metrics
            metrics_file = model_path / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    self.metrics = json.load(f)
            
            self.loaded = True
            logger.info(f"Model {self.model_type.value} loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_type.value}: {str(e)}")
            self.loaded = False
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Make prediction for a single text"""
        if not self.loaded:
            raise ValueError(f"Model {self.model_type.value} is not loaded")
        
        try:
            # Preprocess text
            processed_text = preprocessor.preprocess(text)
            
            # Vectorize
            if self.vectorizer:
                features = self.vectorizer.transform([processed_text])
            else:
                raise ValueError("Vectorizer not loaded")
            
            # Predict
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            
            # Get confidence
            confidence = float(np.max(probabilities))
            
            return {
                "prediction": str(prediction),
                "confidence": confidence,
                "probabilities": probabilities.tolist()
            }
            
        except Exception as e:
            logger.error(f"Prediction failed for {self.model_type.value}: {str(e)}")
            raise


class ModelManager:
    """Manages loading and serving of ML models"""
    
    def __init__(self):
        self.models: Dict[ModelType, ModelMetadata] = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all model instances"""
        for model_type in ModelType:
            self.models[model_type] = ModelMetadata(model_type)
    
    def load_model(self, model_type: ModelType) -> bool:
        """Load a specific model"""
        model = self.models[model_type]
        model.load()
        return model.loaded
    
    def load_all_models(self) -> Dict[ModelType, bool]:
        """Load all available models"""
        results = {}
        for model_type in ModelType:
            results[model_type] = self.load_model(model_type)
        return results
    
    def get_model_info(self, model_type: ModelType) -> Optional[Dict[str, Any]]:
        """Get information about a model"""
        model = self.models.get(model_type)
        if not model or not model.loaded:
            return None
        
        return {
            "name": model.model_type.value,
            "type": model.model_type.value.upper(),
            "status": "loaded" if model.loaded else "not_loaded",
            "accuracy": model.metrics.get("accuracy", 0),
            "precision": model.metrics.get("precision", 0),
            "recall": model.metrics.get("recall", 0),
            "f1_score": model.metrics.get("f1_score", 0),
            "training_date": model.metadata.get("training_date", ""),
            "parameters": model.metadata.get("parameters", {})
        }
    
    def get_all_models_info(self) -> Dict[str, Any]:
        """Get information about all models"""
        info = {}
        for model_type in ModelType:
            model_info = self.get_model_info(model_type)
            if model_info:
                info[model_type.value] = model_info
        return info
    
    def predict(
        self, 
        text: str, 
        model_types: List[ModelType]
    ) -> Dict[str, Any]:
        """Make predictions using specified models"""
        results = []
        
        for model_type in model_types:
            model = self.models.get(model_type)
            if not model or not model.loaded:
                continue
            
            try:
                prediction = model.predict(text)
                results.append({
                    "model": model_type.value,
                    "prediction": prediction["prediction"],
                    "confidence": prediction["confidence"],
                    "probabilities": prediction["probabilities"]
                })
            except Exception as e:
                logger.error(f"Error predicting with {model_type.value}: {e}")
                continue
        
        # Calculate ensemble prediction (simple majority voting)
        if results:
            # Group by prediction
            prediction_counts = {}
            for result in results:
                pred = result["prediction"]
                prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
            
            # Get most common prediction
            ensemble_prediction = max(prediction_counts, key=prediction_counts.get)
            
            # Calculate average confidence for ensemble
            relevant_results = [r for r in results if r["prediction"] == ensemble_prediction]
            ensemble_confidence = sum(r["confidence"] for r in relevant_results) / len(relevant_results)
        else:
            ensemble_prediction = None
            ensemble_confidence = None
        
        return {
            "individual_predictions": results,
            "ensemble_prediction": ensemble_prediction,
            "ensemble_confidence": ensemble_confidence
        }
    
    def ensemble_predict(
        self,
        text: str,
        model_types: List[ModelType],
        method: str = "weighted_voting"
    ) -> Dict[str, Any]:
        """
        Make ensemble predictions using multiple models
        
        Args:
            text: SMS text to classify
            model_types: List of models to use
            method: Ensemble method ('weighted_voting', 'averaging', 'stacking')
        
        Returns:
            Dictionary with ensemble predictions
        """
        predictions = []
        confidences = []
        probabilities_list = []
        
        # Get predictions from each model
        for model_type in model_types:
            model = self.models.get(model_type)
            if not model or not model.loaded:
                continue
            
            try:
                prediction_result = model.predict(text)
                predictions.append(prediction_result["prediction"])
                confidences.append(prediction_result["confidence"])
                probabilities_list.append(prediction_result["probabilities"])
            except Exception as e:
                logger.error(f"Error predicting with {model_type.value}: {e}")
                continue
        
        if not predictions:
            raise ValueError("No valid predictions from selected models")
        
        # Apply ensemble method
        if method == "weighted_voting":
            ensemble_result = self._weighted_voting(
                predictions, confidences, probabilities_list
            )
        elif method == "averaging":
            ensemble_result = self._probability_averaging(
                predictions, probabilities_list
            )
        else:
            # Default to weighted voting
            ensemble_result = self._weighted_voting(
                predictions, confidences, probabilities_list
            )
        
        return {
            "ensemble_prediction": ensemble_result["prediction"],
            "ensemble_confidence": ensemble_result["confidence"],
            "individual_predictions": list(zip(
                [m.value for m in model_types[:len(predictions)]],
                predictions,
                confidences
            ))
        }
    
    def _weighted_voting(
        self,
        predictions: List[str],
        confidences: List[float],
        probabilities_list: List[List[float]]
    ) -> Dict[str, Any]:
        """Weighted voting ensemble method"""
        from collections import defaultdict
        
        # Group predictions with their confidences as weights
        prediction_scores = defaultdict(float)
        
        for pred, confidence in zip(predictions, confidences):
            prediction_scores[pred] += confidence
        
        # Find prediction with highest score
        ensemble_prediction = max(prediction_scores.items(), key=lambda x: x[1])
        
        # Calculate ensemble confidence (normalized score)
        total_score = sum(prediction_scores.values())
        ensemble_confidence = ensemble_prediction[1] / total_score if total_score > 0 else 0
        
        return {
            "prediction": ensemble_prediction[0],
            "confidence": ensemble_confidence,
            "scores": dict(prediction_scores)
        }
    
    def _probability_averaging(
        self,
        predictions: List[str],
        probabilities_list: List[List[float]]
    ) -> Dict[str, Any]:
        """Probability averaging ensemble method"""
        # This assumes all models have same class order
        # In production, you'd need to map classes properly
        
        import numpy as np
        
        # Average probabilities
        avg_probabilities = np.mean(probabilities_list, axis=0)
        
        # Find class with highest average probability
        max_idx = np.argmax(avg_probabilities)
        
        # Get class labels from first model (assuming same order)
        # In practice, you'd need to get this from model metadata
        ensemble_prediction = predictions[0]  # Simplified
        
        return {
            "prediction": ensemble_prediction,
            "confidence": float(avg_probabilities[max_idx]),
            "probabilities": avg_probabilities.tolist()
        }
    
    def compare_models(self, text: str) -> Dict[str, Any]:
        """
        Compare predictions from all loaded models
        
        Args:
            text: SMS text to classify
        
        Returns:
            Comparison results
        """
        results = {}
        
        for model_type, model in self.models.items():
            if not model.loaded:
                continue
            
            try:
                prediction = model.predict(text)
                results[model_type.value] = {
                    "prediction": prediction["prediction"],
                    "confidence": prediction["confidence"],
                    "status": "success"
                }
            except Exception as e:
                results[model_type.value] = {
                    "prediction": "Error",
                    "confidence": 0.0,
                    "status": "error",
                    "error": str(e)
                }
        
        # Calculate agreement
        predictions = [r["prediction"] for r in results.values() 
                      if r["status"] == "success"]
        
        agreement = None
        if predictions:
            unique_predictions = set(predictions)
            agreement = len(unique_predictions) == 1
        
        return {
            "comparison": results,
            "agreement": agreement,
            "total_models": len(results),
            "successful_models": sum(1 for r in results.values() 
                                   if r["status"] == "success")
        }


# Global model manager instance
model_manager = ModelManager()