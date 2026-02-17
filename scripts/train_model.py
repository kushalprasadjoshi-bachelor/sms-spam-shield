#!/usr/bin/env python3
"""
Script to train SMS Spam Shield models.
Updated for Phase 4: LSTM with pre-trained embeddings option
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
from ml.training.trainers.svm_trainer import SVMTrainer
# from ml.training.trainers.lstm_trainer import LSTMTrainer
from backend.app.core.logger import logger


def train_logistic_regression(force_retrain: bool = False):
    """Train Logistic Regression model"""
    logger.info("Starting Logistic Regression training...")
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
        save_training_report("lr", result)
    else:
        logger.error(f"✗ Logistic Regression training failed: {result['error']}")
    return result


def train_naive_bayes(force_retrain: bool = False):
    """Train Naive Bayes model"""
    logger.info("Starting Naive Bayes training...")
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
        save_training_report("nb", result)
    else:
        logger.error(f"✗ Naive Bayes training failed: {result['error']}")
    return result


def train_svm(force_retrain: bool = False, tune: bool = True):
    """Train SVM model with optional hyperparameter tuning"""
    logger.info("Starting SVM training...")
    model_path = Path("models/svm/model.pkl")
    if model_path.exists() and not force_retrain:
        logger.info("✓ SVM model already exists. Use --force to retrain.")
        return {"success": True, "message": "Model already exists"}

    trainer = SVMTrainer(tune_hyperparams=tune)
    result = trainer.train_pipeline()
    if result["success"]:
        logger.info("✓ SVM training completed successfully!")
        logger.info(f"  Accuracy: {result['metrics']['accuracy']:.4f}")
        logger.info(f"  F1-Score: {result['metrics']['f1_score']:.4f}")
        if "best_params" in result:
            logger.info(f"  Best parameters: {result['best_params']}")
        save_training_report("svm", result)
    else:
        logger.error(f"✗ SVM training failed: {result['error']}")
    return result


def train_lstm(force_retrain: bool = False, use_pretrained: bool = False):
    """Train LSTM model with optional pre-trained embeddings"""
    from ml.training.trainers.lstm_trainer import LSTMTrainer
    
    logger.info("Starting LSTM training...")
    model_path = Path("models/lstm/model.h5")
    if model_path.exists() and not force_retrain:
        logger.info("✓ LSTM model already exists. Use --force to retrain.")
        return {"success": True, "message": "Model already exists"}

    trainer = LSTMTrainer(use_pretrained_embeddings=use_pretrained)
    result = trainer.train_pipeline()
    if result["success"]:
        logger.info("✓ LSTM training completed successfully!")
        logger.info(f"  Accuracy: {result['metrics']['accuracy']:.4f}")
        logger.info(f"  F1-Score: {result['metrics']['f1_score']:.4f}")
        save_training_report("lstm", result)
    else:
        logger.error(f"✗ LSTM training failed: {result['error']}")
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
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    logger.info(f"Training report saved: {report_file}")


def compare_models():
    """Compare performance of trained models"""
    logger.info("Comparing model performances...")
    models_info = {}
    for model_type in ["lr", "nb", "svm", "lstm"]:
        metrics_file = Path(f"models/{model_type}/metrics.json")
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                models_info[model_type] = metrics

    if models_info:
        logger.info("Model Comparison:")
        logger.info("-" * 60)
        for model_type, metrics in models_info.items():
            logger.info(f"{model_type.upper():<5} | Accuracy: {metrics.get('accuracy', 0):.4f} | "
                        f"F1-Score: {metrics.get('f1_score', 0):.4f}")
        logger.info("-" * 60)
        best = max(models_info.items(), key=lambda x: x[1].get('accuracy', 0))
        logger.info(f"Best performing model: {best[0].upper()} (Accuracy: {best[1].get('accuracy', 0):.4f})")
    else:
        logger.info("No trained models found.")


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
        "--no-tune",
        action="store_true",
        help="Skip hyperparameter tuning for SVM"
    )
    parser.add_argument(
        "--use-pretrained",
        action="store_true",
        help="Use pre-trained word embeddings for LSTM"
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
        results.append(train_svm(args.force, tune=not args.no_tune))

    if args.model == "lstm" or args.model == "all":
        results.append(train_lstm(args.force, args.use_pretrained))

    if args.compare:
        compare_models()

    failures = [r for r in results if not r.get("success", False)]
    if failures:
        logger.error(f"Training completed with {len(failures)} failures")
        sys.exit(1)
    else:
        logger.info("Training process completed successfully!")


if __name__ == "__main__":
    main()