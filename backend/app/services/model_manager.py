import os
import json
import joblib
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import tensorflow as tf
from enum import Enum

from ..core.config import settings
from ..core.logger import logger
from ..schemas.prediction import ModelType
from .preprocessing import preprocessor


def _load_lstm_trainer():
    from ml.training.trainers.lstm_trainer import LSTMTrainer
    return LSTMTrainer()


MODEL_DIR_ALIASES = {
    "lr": ["lr", "logistic_regression", "logistic"],
    "nb": ["nb", "naive_bayes", "naivebayes"],
    "svm": ["svm"],
    "lstm": ["lstm"],
}


def _resolve_model_dir(model_registry_path: Path, model_key: str) -> Path:
    candidates = MODEL_DIR_ALIASES.get(model_key, [model_key])
    for candidate in candidates:
        candidate_path = model_registry_path / candidate
        if candidate_path.exists():
            return candidate_path
    return model_registry_path / model_key


class ModelMetadata:
    def __init__(self, model_type: ModelType):
        self.model_type = model_type
        self.model = None
        self.vectorizer = None
        self.tokenizer = None
        self.label_encoder = None
        self.metadata = {}
        self.metrics = {}
        self.loaded = False
        self.explainer = None
        self.max_len = 100

    def load(self):
        try:
            model_registry_path = Path(settings.MODEL_REGISTRY_PATH)
            model_path = _resolve_model_dir(model_registry_path, self.model_type.value)
            if model_path.name != self.model_type.value:
                logger.info(
                    f"Resolved model path alias for {self.model_type.value}: {model_path.name}"
                )

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
                    model_file = model_path / "best_model.h5"
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
                model_file = model_path / "model.pkl"
                if model_file.exists():
                    self.model = joblib.load(model_file)

                vectorizer_file = model_path / "vectorizer.pkl"
                if vectorizer_file.exists():
                    self.vectorizer = joblib.load(vectorizer_file)

                encoder_file = model_path / "label_encoder.pkl"
                if encoder_file.exists():
                    self.label_encoder = joblib.load(encoder_file)

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

    def _fallback_tokens_from_text(
        self,
        processed_text: str,
        raw_text: Optional[str] = None,
        max_tokens: int = 10
    ) -> List[Dict[str, float]]:
        source = processed_text.strip() if isinstance(processed_text, str) else ""
        if not source and isinstance(raw_text, str):
            source = raw_text.strip().lower()

        if not source:
            return []

        cleaned = ''.join(ch if ch.isalnum() or ch.isspace() else ' ' for ch in source)
        tokens = [token for token in cleaned.split() if token][:max_tokens]
        if not tokens:
            return []

        base = 1.0
        step = 0.08
        return [
            {"word": token, "importance": max(0.1, base - index * step)}
            for index, token in enumerate(tokens)
        ]

    def _build_token_level_explanation(
        self,
        features,
        processed_text: str,
        raw_text: Optional[str] = None,
        pred_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        if self.vectorizer is None:
            return {
                "method": "token_frequency",
                "important_tokens": self._fallback_tokens_from_text(processed_text, raw_text)
            }

        if not hasattr(self.vectorizer, "get_feature_names_out"):
            return {
                "method": "token_frequency",
                "important_tokens": self._fallback_tokens_from_text(processed_text, raw_text)
            }

        feature_names = self.vectorizer.get_feature_names_out()
        if not hasattr(features, "nonzero"):
            return {
                "method": "token_frequency",
                "important_tokens": self._fallback_tokens_from_text(processed_text, raw_text)
            }

        nonzero_indices = features.nonzero()[1]
        if len(nonzero_indices) == 0:
            return {
                "method": "token_frequency",
                "important_tokens": self._fallback_tokens_from_text(processed_text, raw_text)
            }

        feature_values = features.data if hasattr(features, "data") else np.ones(len(nonzero_indices))
        raw_scores = np.array(feature_values, dtype=float)
        method = "token_weight"

        if hasattr(self.model, "coef_"):
            coef = self.model.coef_
            method = "linear_contribution"
            if coef.ndim == 2:
                if coef.shape[0] == 1:
                    class_coef = coef[0]
                else:
                    class_idx = pred_idx if pred_idx is not None and pred_idx < coef.shape[0] else 0
                    class_coef = coef[class_idx]
                raw_scores = np.array(feature_values, dtype=float) * class_coef[nonzero_indices]
        elif hasattr(self.model, "feature_log_prob_"):
            log_prob = self.model.feature_log_prob_
            method = "naive_bayes_log_prob"
            class_idx = pred_idx if pred_idx is not None and pred_idx < log_prob.shape[0] else 0
            raw_scores = np.array(feature_values, dtype=float) * log_prob[class_idx][nonzero_indices]

        token_scores: Dict[str, float] = {}
        for feature_index, raw_score in zip(nonzero_indices, raw_scores):
            token = str(feature_names[feature_index])
            token_scores[token] = token_scores.get(token, 0.0) + float(abs(raw_score))

        if not token_scores:
            fallback_tokens = self._fallback_tokens_from_text(processed_text, raw_text)
            return {"method": method, "important_tokens": fallback_tokens}

        sorted_tokens = sorted(token_scores.items(), key=lambda item: item[1], reverse=True)[:12]
        max_score = sorted_tokens[0][1] if sorted_tokens else 1.0
        important_tokens = [
            {
                "word": token,
                "importance": (score / max_score) if max_score > 0 else 0.0
            }
            for token, score in sorted_tokens
        ]

        return {
            "method": method,
            "important_tokens": important_tokens
        }

    def predict(self, text: str, include_explanation: bool = False) -> Optional[Dict[str, Any]]:
        if not self.loaded:
            logger.error(f"Model {self.model_type.value} is not loaded")
            return None

        try:
            processed_text = preprocessor.preprocess(text)

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
                    try:
                        attention_model = tf.keras.Model(
                            inputs=self.model.input,
                            outputs=self.model.get_layer('attention_weights').output
                        )
                        att_weights = attention_model.predict(padded, verbose=0)[0]
                        token_words = [self.tokenizer.index_word.get(i, '') for i in seq[0]]
                        important_tokens = []
                        for word, weight in zip(token_words, att_weights[:len(token_words)]):
                            if word and weight > 0.05:
                                important_tokens.append({"word": word, "importance": float(weight)})
                        result["explanation"] = {
                            "method": "attention",
                            "important_tokens": important_tokens[:10],
                            "attention_weights": att_weights.tolist()
                        }
                    except Exception as e:
                        logger.warning(f"Could not extract attention for LSTM: {e}")
                return result

            else:
                if self.vectorizer:
                    features = self.vectorizer.transform([processed_text])
                else:
                    raise ValueError("Vectorizer not loaded")

                if self.model_type.value == "svm" and hasattr(self, 'label_encoder') and self.label_encoder:
                    proba = self.model.predict_proba(features)[0]
                    pred_idx = np.argmax(proba)
                    prediction = self.label_encoder.inverse_transform([pred_idx])[0]
                    confidence = float(proba[pred_idx])
                    probabilities = {
                        self.label_encoder.inverse_transform([i])[0]: float(p)
                        for i, p in enumerate(proba)
                    }

                    result = {
                        "prediction": str(prediction),
                        "confidence": confidence,
                        "probabilities": probabilities
                    }
                    if include_explanation:
                        result["explanation"] = self._build_token_level_explanation(
                            features,
                            processed_text,
                            raw_text=text,
                            pred_idx=int(pred_idx)
                        )
                    return result
            
                else:
                    prediction = self.model.predict(features)[0]
                    proba = self.model.predict_proba(features)[0]
                    confidence = float(np.max(proba))
                    if hasattr(self.model, 'classes_'):
                        probabilities = {
                            str(cls): float(proba[i]) for i, cls in enumerate(self.model.classes_)
                        }
                    else:
                        probabilities = proba.tolist()

                    pred_idx = int(np.argmax(proba)) if len(proba) > 0 else None

                    result = {
                        "prediction": str(prediction),
                        "confidence": confidence,
                        "probabilities": probabilities
                    }
                    if include_explanation:
                        result["explanation"] = self._build_token_level_explanation(
                            features,
                            processed_text,
                            raw_text=text,
                            pred_idx=pred_idx
                        )
                    return result

        except Exception as e:
            logger.error(f"Prediction failed for {self.model_type.value}: {str(e)}")
            return None


class ModelManager:
    def __init__(self):
        self.models: Dict[ModelType, ModelMetadata] = {}
        self._class_order = None
        self._initialize_models()

    def _initialize_models(self):
        for model_type in ModelType:
            self.models[model_type] = ModelMetadata(model_type)

    def load_model(self, model_type: ModelType) -> bool:
        model = self.models[model_type]
        model.load()
        return model.loaded

    def load_all_models(self) -> Dict[ModelType, bool]:
        results = {}
        for model_type in ModelType:
            results[model_type] = self.load_model(model_type)
        return results

    def get_model_info(self, model_type: ModelType) -> Optional[Dict[str, Any]]:
        model = self.models.get(model_type)
        if not model:
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
        info = {}
        for model_type in ModelType:
            model_info = self.get_model_info(model_type)
            if model_info is not None:
                info[model_type.value] = model_info
        return info

    def predict(
        self,
        text: str,
        model_types: List[ModelType],
        include_explanation: bool = False
    ) -> Dict[str, Any]:
        results = []

        for model_type in model_types:
            model = self.models.get(model_type)
            if not model or not model.loaded:
                continue

            prediction = model.predict(text, include_explanation=include_explanation)
            if prediction is None:
                logger.warning(f"Model {model_type.value} returned None, skipping")
                continue

            results.append({
                "model": model_type.value,
                "prediction": prediction["prediction"],
                "confidence": prediction["confidence"],
                "probabilities": prediction.get("probabilities"),
                "explanation": prediction.get("explanation")
            })

        if results:
            prediction_counts = {}
            for result in results:
                pred = result["prediction"]
                prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
            ensemble_prediction = max(prediction_counts, key=prediction_counts.get)
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

    def ensemble_predict(self, text: str, model_types: List[ModelType], method: str = "weighted_voting", include_explanation: bool = False) -> Dict[str, Any]:
        predictions = []
        confidences = []
        probabilities_list = []
        model_names = []
        model_explanations = []

        self._class_order = None

        for model_type in model_types:
            model = self.models.get(model_type)
            if not model or not model.loaded:
                continue

            try:
                pred_result = model.predict(text, include_explanation=include_explanation)
                if pred_result is None:
                    continue
                predictions.append(pred_result["prediction"])
                confidences.append(pred_result["confidence"])
                probs = pred_result.get("probabilities")

                if isinstance(probs, dict):
                    if self._class_order is None:
                        self._class_order = list(probs.keys())
                    probs_list = [probs.get(cls, 0.0) for cls in self._class_order]
                else:
                    if self._class_order is not None and len(self._class_order) == len(probs):
                        probs_list = probs
                    else:
                        probs_list = probs if probs is not None else []
                        if not self._class_order:
                            self._class_order = [str(i) for i in range(len(probs_list))]

                probabilities_list.append(probs_list)
                model_names.append(model_type.value)
                model_explanations.append(pred_result.get("explanation"))
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

        individual = []
        for name, pred, conf, exp in zip(model_names, predictions, confidences, model_explanations):
            individual.append({
                "model": name,
                "prediction": pred,
                "confidence": conf,
                "explanation": exp
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
        if hasattr(self, '_class_order') and self._class_order and max_idx < len(self._class_order):
            pred_label = self._class_order[max_idx]
        else:
            pred_label = predictions[0]
            logger.warning("Class order not set; using first prediction as label")
        return {
            "prediction": pred_label,
            "confidence": float(avg_probs[max_idx]),
            "probabilities": avg_probs.tolist()
        }

    def compare_models(self, text: str) -> Dict[str, Any]:
        results = {}
        for model_type, model in self.models.items():
            if not model.loaded:
                continue
            try:
                pred = model.predict(text, include_explanation=False)
                if pred is None:
                    results[model_type.value] = {
                        "prediction": "Error",
                        "confidence": 0.0,
                        "status": "error",
                        "error": "Prediction returned None"
                    }
                    continue
                params = model.metadata.get("parameters", {})
                results[model_type.value] = {
                    "prediction": pred["prediction"],
                    "confidence": pred["confidence"],
                    "status": "success",
                    "params": params
                }
            except Exception as e:
                results[model_type.value] = {
                    "prediction": "Error",
                    "confidence": 0.0,
                    "status": "error",
                    "error": str(e)
                }

        success_preds = [r["prediction"] for r in results.values() if r["status"] == "success"]
        agreement = len(set(success_preds)) == 1 if success_preds else None

        return {
            "comparison": results,
            "agreement": agreement,
            "total_models": len(results),
            "successful_models": sum(1 for r in results.values() if r["status"] == "success")
        }


model_manager = ModelManager()
