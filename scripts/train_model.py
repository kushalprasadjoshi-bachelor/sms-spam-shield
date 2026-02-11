#!/usr/bin/env python3
"""
Script to train SMS Spam Shield models.
Usage: python train_model.py [model_type]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml.training.trainers.lr_trainer import LogisticRegressionTrainer
from backend.app.core.logger import logger


def train_logistic_regression():
    """Train Logistic Regression model"""
    logger.info("Starting Logistic Regression training...")
    
    trainer = LogisticRegressionTrainer()
    result = trainer.train_pipeline()
    
    if result["success"]:
        logger.info("✓ Logistic Regression training completed successfully!")
        logger.info(f"  Accuracy: {result['metrics']['accuracy']:.4f}")
        logger.info(f"  F1-Score: {result['metrics']['f1_score']:.4f}")
    else:
        logger.error(f"✗ Logistic Regression training failed: {result['error']}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Train SMS Spam Shield models")
    parser.add_argument(
        "model",
        choices=["lr", "nb", "svm", "lstm", "all"],
        help="Model type to train"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining even if model exists"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Training model: {args.model}")
    
    if args.model == "lr" or args.model == "all":
        train_logistic_regression()
    
    # Other models will be added in subsequent phases
    if args.model == "nb":
        logger.info("Naive Bayes training will be implemented in Phase 2")
    elif args.model == "svm":
        logger.info("SVM training will be implemented in Phase 3")
    elif args.model == "lstm":
        logger.info("LSTM training will be implemented in Phase 4")
    
    logger.info("Training process completed")


if __name__ == "__main__":
    main()