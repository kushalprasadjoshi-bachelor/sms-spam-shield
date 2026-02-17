import shap
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.pipeline import Pipeline

from backend.app.core.logger import logger


class ModelExplainer:
    """Provide SHAP-based explanations for model predictions"""
    
    def __init__(self, model, vectorizer, model_type: str, feature_names: Optional[List[str]] = None):
        self.model = model
        self.vectorizer = vectorizer
        self.model_type = model_type
        self.feature_names = feature_names
        self.explainer = None
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize appropriate SHAP explainer based on model type"""
        try:
            if self.model_type in ['lr', 'svm']:
                # For linear models, use LinearExplainer
                self.explainer = shap.LinearExplainer(self.model, masker=self.vectorizer)
            else:
                # For tree-based or kernel models, use KernelExplainer (slower but model-agnostic)
                # For SVM with RBF kernel, we can use KernelExplainer with a background dataset
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba,
                    self._get_background_data()
                )
            logger.info(f"Initialized SHAP explainer for {self.model_type}")
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            self.explainer = None
    
    def _get_background_data(self, n_samples: int = 100):
        """Get background dataset for KernelExplainer"""
        # This should be overridden or passed from training
        # For now, return random samples from the training data
        # In practice, you'd pass a reference to training features
        return np.random.rand(n_samples, self.vectorizer.max_features)
    
    def explain_prediction(self, text: str, features: np.ndarray) -> Dict[str, Any]:
        """Generate SHAP explanation for a single prediction"""
        if self.explainer is None:
            return {"error": "Explainer not initialized"}
        
        try:
            # Get SHAP values
            shap_values = self.explainer.shap_values(features)
            
            # Handle binary vs multi-class
            if isinstance(shap_values, list):
                # Multi-class: shap_values is list of arrays per class
                class_idx = self.model.predict(features)[0]
                shap_values_for_pred = shap_values[class_idx]
            else:
                # Binary: shap_values is array
                shap_values_for_pred = shap_values
            
            # Get feature names
            if hasattr(self.vectorizer, 'get_feature_names_out'):
                feature_names = self.vectorizer.get_feature_names_out()
            else:
                feature_names = [f"feature_{i}" for i in range(shap_values_for_pred.shape[1])]
            
            # Get top contributing words
            importance = shap_values_for_pred[0]
            indices = np.argsort(np.abs(importance))[-10:][::-1]
            
            important_tokens = []
            for idx in indices:
                if importance[idx] != 0:
                    important_tokens.append({
                        "word": feature_names[idx],
                        "importance": float(importance[idx]),
                        "abs_importance": float(np.abs(importance[idx]))
                    })
            
            # Calculate base value (expected value)
            expected_value = self.explainer.expected_value
            if isinstance(expected_value, list):
                expected_value = expected_value[class_idx] if class_idx < len(expected_value) else expected_value[0]
            
            return {
                "method": "shap",
                "base_value": float(expected_value),
                "important_tokens": important_tokens[:5],  # Top 5 tokens
                "shap_values": shap_values_for_pred[0].tolist(),
                "feature_names": feature_names.tolist() if hasattr(feature_names, 'tolist') else feature_names
            }
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return {"error": str(e)}
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get global feature importance for linear models"""
        if self.model_type in ['lr', 'svm'] and hasattr(self.model, 'coef_'):
            coef = self.model.coef_
            if coef.ndim > 1:
                # Multi-class: average absolute coefficients across classes
                coef = np.mean(np.abs(coef), axis=0)
            else:
                coef = np.abs(coef)
            
            # Get feature names
            if hasattr(self.vectorizer, 'get_feature_names_out'):
                feature_names = self.vectorizer.get_feature_names_out()
            else:
                feature_names = [f"feature_{i}" for i in range(len(coef))]
            
            # Create importance dict
            importance = {}
            for name, value in zip(feature_names, coef):
                importance[name] = float(value)
            
            # Sort and return top 50
            sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:50])
            return sorted_importance
        else:
            return {}