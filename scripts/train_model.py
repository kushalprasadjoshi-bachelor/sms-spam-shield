#!/usr/bin/env python3
"""
Script to train SMS Spam Shield models.
Updated for Phase 2: Naive Bayes
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml.training.trainers.lr_trainer import LogisticRegressionTrainer
from ml.training.trainers.nb_trainer import NaiveBayesTrainer
from backend.app.core.logger import logger


def train_logistic_regression(force_retrain: bool = False):
    """Train Logistic Regression model"""
    logger.info("Starting Logistic Regression training...")
    
    # Check if model already exists
    model_path = Path("models/logistic_regression/model.pkl")
    if model_path.exists() and not force_retrain:
        logger.info("✓ Logistic Regression model already exists. Use --force to retrain.")
        return {"success": True, "message": "Model already exists"}
    
    trainer = LogisticRegressionTrainer()
    result = trainer.train_pipeline()
    
    if result["success"]:
        logger.info("✓ Logistic Regression training completed successfully!")
        logger.info(f"  Accuracy: {result['metrics']['accuracy']:.4f}")
        logger.info(f"  F1-Score: {result['metrics']['f1_score']:.4f}")
        
        # Save training report
        save_training_report("lr", result)
    else:
        logger.error(f"✗ Logistic Regression training failed: {result['error']}")
    
    return result


def train_naive_bayes(force_retrain: bool = False):
    """Train Naive Bayes model"""
    logger.info("Starting Naive Bayes training...")
    
    # Check if model already exists
    model_path = Path("models/naive_bayes/model.pkl")
    if model_path.exists() and not force_retrain:
        logger.info("✓ Naive Bayes model already exists. Use --force to retrain.")
        return {"success": True, "message": "Model already exists"}
    
    trainer = NaiveBayesTrainer()
    result = trainer.train_pipeline()
    
    if result["success"]:
        logger.info("✓ Naive Bayes training completed successfully!")
        logger.info(f"  Accuracy: {result['metrics']['accuracy']:.4f}")
        logger.info(f"  F1-Score: {result['metrics']['f1_score']:.4f}")
        
        if "class_priors" in result:
            logger.info(f"  Class Priors: {result['class_priors']}")
        
        # Save training report
        save_training_report("nb", result)
    else:
        logger.error(f"✗ Naive Bayes training failed: {result['error']}")
    
    return result


def save_training_report(model_type: str, result: dict):
    """Save training report for documentation"""
    report_dir = Path("reports/training_reports")
    report_dir.mkdir(exist_ok=True)
    
    report_file = report_dir / f"{model_type}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    report_data = {
        "model": model_type,
        "timestamp": datetime.now().isoformat(),
        "metrics": result.get("metrics", {}),
        "success": result.get("success", False),
        "training_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if "class_priors" in result:
        report_data["class_priors"] = result["class_priors"]
    
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    logger.info(f"Training report saved: {report_file}")


def compare_models():
    """Compare performance of trained models"""
    logger.info("Comparing model performances...")
    
    models_info = {}
    
    for model_type in ["lr", "nb"]:
        metrics_file = Path(f"models/{model_type}/metrics.json")
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                models_info[model_type] = metrics
    
    if models_info:
        logger.info("Model Comparison:")
        logger.info("-" * 50)
        for model_type, metrics in models_info.items():
            logger.info(f"{model_type.upper():<5} | Accuracy: {metrics.get('accuracy', 0):.4f} | "
                       f"F1-Score: {metrics.get('f1_score', 0):.4f}")
        logger.info("-" * 50)
        
        # Determine best model
        best_model = max(models_info.items(), key=lambda x: x[1].get('accuracy', 0))
        logger.info(f"Best performing model: {best_model[0].upper()} "
                   f"(Accuracy: {best_model[1].get('accuracy', 0):.4f})")


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
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare model performances after training"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Training model: {args.model}")
    logger.info(f"Force retrain: {args.force}")
    
    results = []
    
    if args.model == "lr" or args.model == "all":
        results.append(train_logistic_regression(args.force))
    
    if args.model == "nb" or args.model == "all":
        results.append(train_naive_bayes(args.force))
    
    if args.model == "svm" or args.model == "all":
        logger.info("SVM training will be implemented in Phase 3")
    
    if args.model == "lstm" or args.model == "all":
        logger.info("LSTM training will be implemented in Phase 4")
    
    # Compare models if requested
    if args.compare:
        compare_models()
    
    # Check for failures
    failures = [r for r in results if not r.get("success", False)]
    if failures:
        logger.error(f"Training completed with {len(failures)} failures")
        sys.exit(1)
    else:
        logger.info("Training process completed successfully!")


if __name__ == "__main__":
    main()