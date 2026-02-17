import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any, Optional, Tuple
import joblib
import json
from pathlib import Path

from .base_trainer import BaseTrainer
from backend.app.core.config import settings
from backend.app.core.logger import logger
from ml.training.versioning import ModelVersionManager


class SVMTrainer(BaseTrainer):
    """Support Vector Machine trainer with hyperparameter tuning and versioning"""
    
    def __init__(self, tune_hyperparams: bool = True):
        super().__init__("svm")
        self.tune_hyperparams = tune_hyperparams
        self.version_manager = ModelVersionManager(self.model_name)
        self.label_encoder = LabelEncoder()
        self.best_params = {}
        self.cv_results = {}
        
        # Default parameters (will be overridden if tuning)
        self.model_params = {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "probability": True,
            "random_state": settings.RANDOM_STATE,
            "class_weight": "balanced"
        }
    
    def create_vectorizer(self):
        """Create TF-IDF vectorizer with n-grams"""
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),  # Use trigrams for SVM
            stop_words='english',
            sublinear_tf=True,  # Use 1+log(tf)
            use_idf=True
        )
    
    def create_model(self):
        """Create base SVM model"""
        self.model = SVC(**self.model_params)
    
    def tune_hyperparameters(self, X_train_features, y_train):
        """Perform hyperparameter tuning using GridSearchCV"""
        logger.info("Starting SVM hyperparameter tuning...")
        
        # Define parameter grid
        param_grid = [
            {
                'kernel': ['linear'],
                'C': [0.1, 1.0, 10.0, 100.0],
                'class_weight': ['balanced', None]
            },
            {
                'kernel': ['rbf'],
                'C': [0.1, 1.0, 10.0, 100.0],
                'gamma': ['scale', 'auto', 0.01, 0.1, 1.0],
                'class_weight': ['balanced', None]
            },
            {
                'kernel': ['poly'],
                'C': [0.1, 1.0, 10.0],
                'degree': [2, 3, 4],
                'gamma': ['scale', 'auto'],
                'class_weight': ['balanced', None]
            }
        ]
        
        # Stratified K-Fold cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=settings.RANDOM_STATE)
        
        # Grid search with probability=True (requires more computation)
        grid_search = GridSearchCV(
            SVC(probability=True, random_state=settings.RANDOM_STATE),
            param_grid,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X_train_features, y_train)
        
        # Store best parameters and results
        self.best_params = grid_search.best_params_
        self.cv_results = {
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_,
            'cv_results_': {k: str(v) for k, v in grid_search.cv_results_.items() 
                           if k not in ['params']}
        }
        
        logger.info(f"Best SVM parameters: {self.best_params}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Return best estimator
        return grid_search.best_estimator_
    
    def train(self):
        """Train SVM model with optional hyperparameter tuning"""
        logger.info("Training SVM model...")
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(self.y_train)
        self.y_test_encoded = self.label_encoder.transform(self.y_test)
        
        if self.tune_hyperparams:
            # Perform hyperparameter tuning
            self.model = self.tune_hyperparameters(self.X_train_features, y_train_encoded)
            # Update model_params with best params
            self.model_params.update(self.model.get_params())
        else:
            # Train with default parameters
            self.model.fit(self.X_train_features, y_train_encoded)
        
        logger.info("SVM training completed")
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate SVM model with encoded labels"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        # Make predictions
        y_pred = self.model.predict(self.X_test_features)
        
        # Decode predictions back to original labels
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        y_test_labels = self.y_test  # Original test labels
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_labels, y_pred_labels)
        precision = precision_score(y_test_labels, y_pred_labels, average='weighted')
        recall = recall_score(y_test_labels, y_pred_labels, average='weighted')
        f1 = f1_score(y_test_labels, y_pred_labels, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_test_labels, y_pred_labels, 
                             labels=self.label_encoder.classes_)
        
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist(),
            "classes": self.label_encoder.classes_.tolist()
        }
        
        if self.tune_hyperparams:
            metrics.update({
                "best_cv_score": float(self.cv_results['best_score']),
                "best_params": self.cv_results['best_params']
            })
        
        logger.info(f"SVM Accuracy: {accuracy:.4f}")
        logger.info(f"SVM F1-Score: {f1:.4f}")
        
        return metrics
    
    def save_model(self, metrics: Dict[str, float], params: Dict[str, Any]):
        """Save model with versioning"""
        # Save label encoder
        encoder_path = self.version_manager.model_dir / "label_encoder.pkl"
        joblib.dump(self.label_encoder, encoder_path)
        
        # Save hyperparameter tuning results if available
        if self.cv_results:
            cv_results_path = self.version_manager.model_dir / "cv_results.json"
            with open(cv_results_path, "w") as f:
                json.dump(self.cv_results, f, indent=2)
        
        # Save version
        version = self.version_manager.save_version(
            model=self.model,
            vectorizer=self.vectorizer,
            metrics=metrics,
            params=params
        )
        
        # Set as production if it's the first version
        if len(self.version_manager.list_versions()) == 1:
            self.version_manager.set_production(version)
    
    def train_pipeline(self) -> Dict[str, Any]:
        """Complete training pipeline for SVM"""
        try:
            # Load data
            df = self.load_data()
            
            # Preprocess
            X, y = self.preprocess_data(df)
            
            # Split data
            self.split_data(X, y)
            
            # Create vectorizer
            self.create_vectorizer()
            
            # Extract features
            self.extract_features(self.X_train, self.X_test)
            
            # Train model (includes tuning if enabled)
            self.train()
            
            # Evaluate
            metrics = self.evaluate()
            
            # Save model
            self.save_model(metrics, self.model_params)
            
            return {
                "success": True,
                "metrics": metrics,
                "model": self.model_name,
                "best_params": self.best_params,
                "cv_results": self.cv_results
            }
            
        except Exception as e:
            logger.error(f"SVM training failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "model": self.model_name
            }
    
    def load_production_model(self):
        """Load the production version of the model"""
        try:
            model, vectorizer, metadata = self.version_manager.load_version(
                self.version_manager.get_production_version()["version"]
            )
            self.model = model
            self.vectorizer = vectorizer
            self.label_encoder = joblib.load(
                self.version_manager.model_dir / "label_encoder.pkl"
            )
            logger.info(f"Loaded production SVM model (version {metadata['version']})")
            return True
        except Exception as e:
            logger.error(f"Failed to load production SVM model: {e}")
            return False