import abc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any, Tuple
import joblib
import json
from datetime import datetime
from pathlib import Path

from backend.app.core.config import settings
from backend.app.core.logger import logger
from backend.app.services.preprocessing import preprocessor


class BaseTrainer(abc.ABC):
    """Base class for all model trainers"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.vectorizer = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare dataset"""
        logger.info(f"Loading dataset from {settings.DATASET_PATH}")
        
        try:
            df = pd.read_csv(settings.DATASET_PATH)
            logger.info(f"Dataset loaded: {len(df)} samples")
            
            # Ensure required columns exist
            if 'message_text' not in df.columns or 'category' not in df.columns:
                raise ValueError("Dataset must contain 'message_text' and 'category' columns")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess the dataset"""
        logger.info("Preprocessing dataset...")
        
        # Clean text
        df['cleaned_text'] = preprocessor.preprocess_batch(df['message_text'].astype(str))
        
        # Remove empty texts
        df = df[df['cleaned_text'].str.len() > 0]
        
        # Separate features and labels
        X = df['cleaned_text']
        y = df['category']
        
        logger.info(f"After preprocessing: {len(X)} samples")
        
        return X, y
    
    def split_data(self, X: pd.Series, y: pd.Series) -> None:
        """Split data into train and test sets"""
        logger.info("Splitting data into train/test sets...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=settings.TEST_SIZE,
            random_state=settings.RANDOM_STATE,
            stratify=y
        )
        
        logger.info(f"Training set: {len(self.X_train)} samples")
        logger.info(f"Testing set: {len(self.X_test)} samples")
    
    @abc.abstractmethod
    def create_vectorizer(self):
        """Create feature vectorizer"""
        pass
    
    @abc.abstractmethod
    def create_model(self):
        """Create the ML model"""
        pass
    
    def extract_features(self, X_train, X_test):
        """Extract features using vectorizer"""
        logger.info("Extracting features...")
        
        self.X_train_features = self.vectorizer.fit_transform(X_train)
        self.X_test_features = self.vectorizer.transform(X_test)
        
        logger.info(f"Feature dimensions: {self.X_train_features.shape}")
    
    @abc.abstractmethod
    def train(self):
        """Train the model"""
        pass
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model performance"""
        logger.info("Evaluating model...")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test_features)
        y_pred_proba = self.model.predict_proba(self.X_test_features)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        
        return metrics
    
    def save_model(self, metrics: Dict[str, float], params: Dict[str, Any]):
        """Save trained model and metadata"""
        logger.info(f"Saving {self.model_name} model...")
        
        # Create model directory
        model_dir = Path(settings.MODEL_REGISTRY_PATH) / self.model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / "model.pkl"
        joblib.dump(self.model, model_path)
        
        # Save vectorizer
        vectorizer_path = model_dir / "vectorizer.pkl"
        joblib.dump(self.vectorizer, vectorizer_path)
        
        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "training_date": datetime.now().isoformat(),
            "n_samples": len(self.X_train),
            "feature_dimensions": self.X_train_features.shape[1],
            "parameters": params,
            "classes": self.model.classes_.tolist()
        }
        
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save metrics
        metrics_path = model_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Model saved to {model_dir}")
    
    def train_pipeline(self) -> Dict[str, Any]:
        """Complete training pipeline"""
        try:
            # Load data
            df = self.load_data()
            
            # Preprocess
            X, y = self.preprocess_data(df)
            
            # Split data
            self.split_data(X, y)
            
            # Create vectorizer and model
            self.create_vectorizer()
            self.create_model()
            
            # Extract features
            self.extract_features(self.X_train, self.X_test)
            
            # Train model
            self.train()
            
            # Evaluate
            metrics = self.evaluate()
            
            return {
                "success": True,
                "metrics": metrics,
                "model": self.model_name
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "model": self.model_name
            }