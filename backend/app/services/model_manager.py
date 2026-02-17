import os
import json
import joblib
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import tensorflow as tf
from enum import Enum  # <-- Added missing import

from backend.app.core.config import settings
from backend.app.core.logger import logger
from backend.app.schemas.prediction import ModelType
from backend.app.services.preprocessing import preprocessor


# Conditional import for LSTM trainer (avoid circular import)
def _load_lstm_trainer():
    from ml.training.trainers.lstm_trainer import LSTMTrainer
    return LSTMTrainer()


# ModelType is already defined in schemas; but if needed locally:
# class ModelType(str, Enum):
#     LOGISTIC_REGRESSION = "lr"
#     NAIVE_BAYES = "nb"
#     SVM = "svm"
#     LSTM = "lstm"


class ModelMetadata:
    def __init__(self, model_type: ModelType):
        self.model_type = model_type
        self.model = None
        self.vectorizer = None      # for classical models
        self.tokenizer = None        # for LSTM
        self.label_encoder = None    # for LSTM/SVM
        self.metadata = {}
        self.metrics = {}
        self.loaded = False
        self.explainer = None
        self.max_len = 100            # for LSTM

    def load(self):
        """Load model from disk – supports classical and LSTM models."""
        try:
            model_path = Path(settings.MODEL_REGISTRY_PATH) / self.model_type.value

            # Special handling for LSTM
            if self.model_type.value == "lstm":
                trainer = _load_lstm_trainer()
                if trainer.load_production_model():
                    self.model = trainer.model
                    self.tokenizer = trainer.tokenizer
                    self.label_encoder = trainer.label_encoder
                    self.metadata = trainer.version_manager.get_production_version() or {}
                    self.metrics = self.metadata.get("metrics", {})
                    self.max_len = trainer.max_len
                    self.loaded = True
                    logger.info("LSTM model loaded successfully")
                else:
                    logger.warning("No production LSTM model found, trying legacy loading...")
                    # Fallback: try to load legacy files (if any)
                    model_file = model_path / "model.h5"
                    if model_file.exists():
                        self.model = tf.keras.models.load_model(model_file)
                    tokenizer_file = model_path / "tokenizer.pkl"
                    if tokenizer_file.exists():
                        import pickle
                        with open(tokenizer_file, 'rb') as f:
                            self.tokenizer = pickle.load(f)
                    encoder_file = model_path / "label_encoder.pkl"
                    if encoder_file.exists():
                        with open(encoder_file, 'rb') as f:
                            self.label_encoder = pickle.load(f)
                    if self.model and self.tokenizer and self.label_encoder:
                        self.loaded = True
                        logger.info("LSTM loaded from legacy files")
                    else:
                        self.loaded = False

            else:
                # Classical models (LR, NB, SVM)
                model_file = model_path / "model.pkl"
                if model_file.exists():
                    self.model = joblib.load(model_file)

                vectorizer_file = model_path / "vectorizer.pkl"
                if vectorizer_file.exists():
                    self.vectorizer = joblib.load(vectorizer_file)

                # For SVM, also load label encoder if exists
                encoder_file = model_path / "label_encoder.pkl"
                if encoder_file.exists():
                    with open(encoder_file, 'rb') as f:
                        self.label_encoder = joblib.load(f)   # or pickle?

                metadata_file = model_path / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        self.metadata = json.load(f)

                metrics_file = model_path / "metrics.json"
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        self.metrics = json.load(f)

                self.loaded = self.model is not None and self.vectorizer is not None

            if self.loaded:
                logger.info(f"Model {self.model_type.value} loaded successfully")
            else:
                logger.warning(f"Model {self.model_type.value} could not be loaded")

        except Exception as e:
            logger.error(f"Failed to load model {self.model_type.value}: {str(e)}")
            self.loaded = False

    def predict(self, text: str, include_explanation: bool = False) -> Dict[str, Any]:
        """Make prediction for a single text, optionally returning explanation."""
        if not self.loaded:
            raise ValueError(f"Model {self.model_type.value} is not loaded")

        try:
            processed_text = preprocessor.preprocess(text)

            # LSTM branch
            if self.model_type.value == "lstm":
                seq = self.tokenizer.texts_to_sequences([processed_text])
                padded = tf.keras.preprocessing.sequence.pad_sequences(
                    seq, maxlen=self.max_len, padding='post', truncating='post'
                )
                proba = self.model.predict(padded, verbose=0)[0]
                pred_idx = np.argmax(proba)
                prediction = self.label_encoder.inverse_transform([pred_idx])[0]
                confidence = float(proba[pred_idx])

                result = {
                    "prediction": str(prediction),
                    "confidence": confidence,
                    "probabilities": {
                        self.label_encoder.inverse_transform([i])[0]: float(p)
                        for i, p in enumerate(proba)
                    }
                }

                if include_explanation:
                    # Extract attention weights
                    try:
                        # Get attention layer output
                        attention_model = tf.keras.Model(
                            inputs=self.model.input,
                            outputs=self.model.get_layer('attention_weights').output
                        )
                        att_weights = attention_model.predict(padded, verbose=0)[0]
                        # Map tokens to words
                        token_words = [self.tokenizer.index_word.get(i, '') for i in seq[0]]
                        important_tokens = []
                        for word, weight in zip(token_words, att_weights[:len(token_words)]):
                            if word and weight > 0.05:  # threshold
                                important_tokens.append({
                                    "word": word,
                                    "importance": float(weight)
                                })
                        result["explanation"] = {
                            "method": "attention",
                            "important_tokens": important_tokens[:10],
                            "attention_weights": att_weights.tolist()
                        }
                    except Exception as e:
                        logger.warning(f"Could not extract attention: {e}")
                        result["explanation"] = {"method": "attention", "error": str(e)}
                return result

            # Classical models
            else:
                if self.vectorizer:
                    features = self.vectorizer.transform([processed_text])
                else:
                    raise ValueError("Vectorizer not loaded")

                if self.model_type.value == "svm" and hasattr(self, 'label_encoder') and self.label_encoder:
                    # SVM with probability=True
                    proba = self.model.predict_proba(features)[0]
                    pred_idx = np.argmax(proba)
                    prediction = self.label_encoder.inverse_transform([pred_idx])[0]
                    confidence = float(proba[pred_idx])
                    probabilities = {
                        self.label_encoder.inverse_transform([i])[0]: float(p)
                        for i, p in enumerate(proba)
                    }
                else:
                    # LR, NB
                    prediction = self.model.predict(features)[0]
                    proba = self.model.predict_proba(features)[0]
                    confidence = float(np.max(proba))
                    # Build class-probability mapping
                    if hasattr(self.model, 'classes_'):
                        probabilities = {
                            str(cls): float(proba[i]) for i, cls in enumerate(self.model.classes_)
                        }
                    else:
                        probabilities = proba.tolist()

                result = {
                    "prediction": str(prediction),
                    "confidence": confidence,
                    "probabilities": probabilities
                }

                if include_explanation:
                    # For classical models we can use simple token importance (or SHAP if available)
                    # For now, use extract_important_tokens (as before)
                    # (We might integrate SHAP later)
                    pass

                return result

        except Exception as e:
            logger.error(f"Prediction failed for {self.model_type.value}: {str(e)}")
            raise


class ModelManager:
    """Manages loading and serving of ML models"""

    def __init__(self):
        self.models: Dict[ModelType, ModelMetadata] = {}
        self._class_order = None  # Will be set during ensemble prediction
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
        model_types: List[ModelType],
        include_explanation: bool = False
    ) -> Dict[str, Any]:
        """Make predictions using specified models (simple majority voting)."""
        results = []

        for model_type in model_types:
            model = self.models.get(model_type)
            if not model or not model.loaded:
                continue

            try:
                prediction = model.predict(text, include_explanation=include_explanation)
                results.append({
                    "model": model_type.value,
                    "prediction": prediction["prediction"],
                    "confidence": prediction["confidence"],
                    "probabilities": prediction.get("probabilities"),
                    "explanation": prediction.get("explanation")
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

    # ---------- Advanced ensemble methods ----------
    def ensemble_predict(
        self,
        text: str,
        model_types: List[ModelType],
        method: str = "weighted_voting",
        include_explanation: bool = False
    ) -> Dict[str, Any]:
        """
        Make ensemble predictions using multiple models with advanced methods.
        """
        predictions = []
        confidences = []
        probabilities_list = []
        model_names = []

        # Reset class order for this call
        self._class_order = None

        for model_type in model_types:
            model = self.models.get(model_type)
            if not model or not model.loaded:
                continue

            try:
                pred_result = model.predict(text, include_explanation=include_explanation)
                predictions.append(pred_result["prediction"])
                confidences.append(pred_result["confidence"])
                probs = pred_result.get("probabilities")

                # Ensure probabilities are a dictionary with class names as keys
                if isinstance(probs, dict):
                    # Set class order from the first successful model that returns a dict
                    if self._class_order is None:
                        self._class_order = list(probs.keys())
                    # Convert to list in the established order
                    probs_list = [probs.get(cls, 0.0) for cls in self._class_order]
                else:
                    # If probs is a list, try to map using class_order if available
                    if self._class_order is not None and len(self._class_order) == len(probs):
                        probs_list = probs
                    else:
                        # Fallback: assume probs list is in correct order (should not happen)
                        probs_list = probs if probs is not None else []
                        if not self._class_order:
                            # Create a dummy class order from indices (only for list)
                            self._class_order = [str(i) for i in range(len(probs_list))]

                probabilities_list.append(probs_list)
                model_names.append(model_type.value)
            except Exception as e:
                logger.error(f"Error predicting with {model_type.value}: {e}")
                continue

        if not predictions:
            raise ValueError("No valid predictions from selected models")

        if method == "weighted_voting":
            ensemble_result = self._weighted_voting(predictions, confidences, probabilities_list)
        elif method == "averaging":
            ensemble_result = self._probability_averaging(predictions, probabilities_list)
        else:
            ensemble_result = self._weighted_voting(predictions, confidences, probabilities_list)

        # Build individual results list
        individual = []
        for name, pred, conf in zip(model_names, predictions, confidences):
            individual.append({
                "model": name,
                "prediction": pred,
                "confidence": conf
            })

        return {
            "ensemble_prediction": ensemble_result["prediction"],
            "ensemble_confidence": ensemble_result["confidence"],
            "individual_predictions": individual
        }

    def _weighted_voting(self, predictions, confidences, probabilities_list):
        from collections import defaultdict
        scores = defaultdict(float)
        for pred, conf in zip(predictions, confidences):
            scores[pred] += conf
        best_pred = max(scores.items(), key=lambda x: x[1])
        total = sum(scores.values())
        conf = best_pred[1] / total if total > 0 else 0
        return {"prediction": best_pred[0], "confidence": conf}

    def _probability_averaging(self, predictions, probabilities_list):
        import numpy as np
        avg_probs = np.mean(probabilities_list, axis=0)
        max_idx = np.argmax(avg_probs)
        # Use stored class order if available
        if hasattr(self, '_class_order') and self._class_order and max_idx < len(self._class_order):
            pred_label = self._class_order[max_idx]
        else:
            # Fallback: use first prediction (less accurate but safe)
            pred_label = predictions[0]
            logger.warning("Class order not set; using first prediction as label")
        return {
            "prediction": pred_label,
            "confidence": float(avg_probs[max_idx]),
            "probabilities": avg_probs.tolist()
        }

    def compare_models(self, text: str) -> Dict[str, Any]:
        """Compare predictions from all loaded models."""
        results = {}
        for model_type, model in self.models.items():
            if not model.loaded:
                continue
            try:
                pred = model.predict(text, include_explanation=False)
                params = model.metadata.get("parameters", {})  # Get stored parameters
                results[model_type.value] = {
                    "prediction": pred["prediction"],
                    "confidence": pred["confidence"],
                    "status": "success",
                    "params": params  # Add parameters
                }
            except Exception as e:
                results[model_type.value] = {
                    "prediction": "Error",
                    "confidence": 0.0,
                    "status": "error",
                    "error": str(e)
                }
        # Calculate agreement
        success_preds = [r["prediction"] for r in results.values() if r["status"] == "success"]
        agreement = len(set(success_preds)) == 1 if success_preds else None

        return {
            "comparison": results,
            "agreement": agreement,
            "total_models": len(results),
            "successful_models": sum(1 for r in results.values() if r["status"] == "success")
        }


# Global model manager instance
model_manager = ModelManager()