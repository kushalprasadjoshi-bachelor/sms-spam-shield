from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from typing import Dict, Any
import numpy as np

from .base_trainer import BaseTrainer
from backend.app.core.logger import logger


class NaiveBayesTrainer(BaseTrainer):
    """Naive Bayes trainer with Multinomial Naive Bayes"""
    
    def __init__(self):
        super().__init__("naive_bayes")
        self.model_params = {
            "alpha": 1.0,  # Laplace smoothing parameter
            "fit_prior": True,  # Learn class prior probabilities
            "class_prior": None  # Prior probabilities of classes
        }
    
    def create_vectorizer(self):
        """Create Count Vectorizer for Naive Bayes"""
        # Naive Bayes typically uses CountVectorizer instead of TF-IDF
        self.vectorizer = CountVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # Use unigrams and bigrams
            stop_words='english',
            binary=False  # Use word counts (not binary)
        )
    
    def create_model(self):
        """Create Multinomial Naive Bayes model"""
        self.model = MultinomialNB(**self.model_params)
    
    def train(self):
        """Train the model"""
        logger.info("Training Naive Bayes model...")
        self.model.fit(self.X_train_features, self.y_train)
        logger.info("Naive Bayes training completed")
    
    def calculate_class_priors(self, y: np.ndarray) -> Dict[str, float]:
        """Calculate and return class prior probabilities"""
        unique_classes, class_counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        priors = {str(cls): count/total_samples for cls, count in zip(unique_classes, class_counts)}
        
        logger.info(f"Class priors: {priors}")
        return priors
    
    def train_pipeline(self) -> Dict[str, Any]:
        """Run complete training pipeline with Naive Bayes specific steps"""
        try:
            # Load data
            df = self.load_data()
            
            # Preprocess
            X, y = self.preprocess_data(df)
            
            # Split data
            self.split_data(X, y)
            
            # Calculate class priors
            class_priors = self.calculate_class_priors(self.y_train)
            self.model_params["class_prior"] = list(class_priors.values())
            
            # Create vectorizer and model
            self.create_vectorizer()
            self.create_model()
            
            # Extract features
            self.extract_features(self.X_train, self.X_test)
            
            # Train model
            self.train()
            
            # Evaluate
            metrics = self.evaluate()
            
            # Save model with priors
            self.save_model(metrics, self.model_params)
            
            return {
                "success": True,
                "metrics": metrics,
                "model": self.model_name,
                "class_priors": class_priors
            }
            
        except Exception as e:
            logger.error(f"Naive Bayes training failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "model": self.model_name
            }
    
    def evaluate(self) -> Dict[str, float]:
        """Extended evaluation with additional Naive Bayes metrics"""
        # Get standard metrics from parent class
        metrics = super().evaluate()
        
        # Calculate additional Naive Bayes specific metrics
        try:
            # Log probability of features given classes
            feature_log_probs = self.model.feature_log_prob_
            logger.info(f"Feature log probabilities shape: {feature_log_probs.shape}")
            
            # Calculate class log priors
            class_log_priors = self.model.class_log_prior_
            logger.info(f"Class log priors: {class_log_priors}")
            
            # Add to metrics
            metrics["log_likelihood"] = float(np.sum(self.model.feature_log_prob_))
            
        except Exception as e:
            logger.error(f"Could not calculate Naive Bayes specific metrics: {str(e)}")
        
        return metrics