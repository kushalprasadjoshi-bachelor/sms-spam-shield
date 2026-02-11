from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, Any

from .base_trainer import BaseTrainer
from backend.app.core.logger import logger


class LogisticRegressionTrainer(BaseTrainer):
    """Logistic Regression trainer"""
    
    def __init__(self):
        super().__init__("logistic_regression")
        self.model_params = {
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
            "multi_class": "multinomial",
            "random_state": 42
        }
    
    def create_vectorizer(self):
        """Create TF-IDF vectorizer"""
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # Use unigrams and bigrams
            stop_words='english'
        )
    
    def create_model(self):
        """Create Logistic Regression model"""
        self.model = LogisticRegression(**self.model_params)
    
    def train(self):
        """Train the model"""
        logger.info("Training Logistic Regression model...")
        self.model.fit(self.X_train_features, self.y_train)
        logger.info("Training completed")
    
    def train_pipeline(self) -> Dict[str, Any]:
        """Run complete training pipeline"""
        result = super().train_pipeline()
        
        if result["success"]:
            # Save model with specific parameters
            self.save_model(result["metrics"], self.model_params)
        
        return result