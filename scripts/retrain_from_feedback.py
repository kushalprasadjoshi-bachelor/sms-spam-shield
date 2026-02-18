#!/usr/bin/env python3
"""
Retrain SMS Spam Shield models using stored user feedback.

Workflow:
1. Load stored feedback from feedback.json.
2. Augment the base dataset with corrected samples.
3. Retrain models using existing trainer classes.
4. Save new model versions using the current versioning system.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import settings
from backend.app.core.logger import logger
from backend.app.services.feedback_service import feedback_service
from ml.training.trainers.lr_trainer import LogisticRegressionTrainer
from ml.training.trainers.nb_trainer import NaiveBayesTrainer
from ml.training.trainers.svm_trainer import SVMTrainer
from ml.training.trainers.lstm_trainer import LSTMTrainer
from ml.training.versioning import ModelVersionManager


def load_feedback_rows() -> List[Dict[str, str]]:
    """Convert feedback records into dataset rows."""
    rows: List[Dict[str, str]] = []
    for record in feedback_service.list_feedback():
        sms_text = str(record.sms or "").strip()
        corrected_label = str(record.corrected_label or "").strip().lower()
        if not sms_text or not corrected_label:
            continue
        rows.append({
            "message_text": sms_text,
            "category": corrected_label
        })
    return rows


def build_augmented_dataset(
    base_dataset_path: Path,
    feedback_rows: List[Dict[str, str]],
    output_dataset_path: Path
) -> Tuple[int, int, int]:
    """Create an augmented dataset where feedback rows override duplicates by message text."""
    base_df = pd.read_csv(base_dataset_path)
    if "message_text" not in base_df.columns or "category" not in base_df.columns:
        raise ValueError("Base dataset must contain 'message_text' and 'category' columns")

    feedback_df = pd.DataFrame(feedback_rows)
    if feedback_df.empty:
        raise ValueError("No usable feedback rows found")

    # Keep feedback rows as higher priority when the same message appears in base data.
    base_df = base_df[["message_text", "category"]].copy()
    base_df["__priority"] = 0
    feedback_df = feedback_df[["message_text", "category"]].copy()
    feedback_df["__priority"] = 1

    combined = pd.concat([base_df, feedback_df], ignore_index=True)
    combined["message_text"] = combined["message_text"].astype(str).str.strip()
    combined["category"] = combined["category"].astype(str).str.strip().str.lower()
    combined = combined[(combined["message_text"] != "") & (combined["category"] != "")]
    combined = combined.sort_values("__priority").drop_duplicates(subset=["message_text"], keep="last")
    combined = combined.drop(columns=["__priority"])

    output_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_dataset_path, index=False)
    return len(base_df), len(feedback_df), len(combined)


def latest_new_version(model_name: str, before_versions: set) -> str | None:
    """Return the latest version created after training."""
    manager = ModelVersionManager(model_name)
    after_versions = manager.list_versions()
    for version_info in after_versions:
        version_name = version_info.get("version")
        if version_name and version_name not in before_versions:
            return version_name
    return None


def train_lr(set_production: bool) -> Dict[str, object]:
    trainer = LogisticRegressionTrainer()
    result = trainer.train_pipeline()
    if not result.get("success"):
        return result

    version_manager = ModelVersionManager("logistic_regression")
    version = version_manager.save_version(
        model=trainer.model,
        vectorizer=trainer.vectorizer,
        metrics=result.get("metrics", {}),
        params=trainer.model_params
    )
    if set_production:
        version_manager.set_production(version)

    result["version"] = version
    return result


def train_nb(set_production: bool) -> Dict[str, object]:
    trainer = NaiveBayesTrainer()
    result = trainer.train_pipeline()
    if not result.get("success"):
        return result

    version_manager = ModelVersionManager("naive_bayes")
    version = version_manager.save_version(
        model=trainer.model,
        vectorizer=trainer.vectorizer,
        metrics=result.get("metrics", {}),
        params=trainer.model_params
    )
    if set_production:
        version_manager.set_production(version)

    result["version"] = version
    return result


def train_svm(set_production: bool, tune_svm: bool) -> Dict[str, object]:
    version_manager = ModelVersionManager("svm")
    before_versions = {item["version"] for item in version_manager.list_versions()}

    trainer = SVMTrainer(tune_hyperparams=tune_svm)
    result = trainer.train_pipeline()
    if not result.get("success"):
        return result

    new_version = latest_new_version("svm", before_versions)
    if set_production and new_version:
        version_manager.set_production(new_version)

    result["version"] = new_version
    return result


def train_lstm() -> Dict[str, object]:
    version_manager = ModelVersionManager("lstm")
    before_versions = {item["version"] for item in version_manager.list_versions()}

    trainer = LSTMTrainer(use_pretrained_embeddings=False)
    result = trainer.train_pipeline()
    if not result.get("success"):
        return result

    result["version"] = latest_new_version("lstm", before_versions)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrain models using feedback-corrected labels.")
    parser.add_argument("--min-feedback", type=int, default=1, help="Minimum feedback records required to retrain")
    parser.add_argument("--set-production", action="store_true", help="Set newly trained LR/NB/SVM versions as production")
    parser.add_argument("--no-tune-svm", action="store_true", help="Disable SVM hyperparameter tuning for faster training")
    parser.add_argument("--skip-lstm", action="store_true", help="Skip LSTM retraining")
    args = parser.parse_args()

    feedback_rows = load_feedback_rows()
    if len(feedback_rows) < args.min_feedback:
        logger.warning(
            f"Not enough feedback to retrain. Found {len(feedback_rows)} records, require {args.min_feedback}."
        )
        return 1

    base_dataset_path = Path(settings.DATASET_PATH)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    augmented_dataset_path = (
        PROJECT_ROOT / "dataset" / "processed" / f"processed_dataset_feedback_augmented_{timestamp}.csv"
    )

    base_count, feedback_count, final_count = build_augmented_dataset(
        base_dataset_path=base_dataset_path,
        feedback_rows=feedback_rows,
        output_dataset_path=augmented_dataset_path
    )
    logger.info(
        f"Augmented dataset saved: {augmented_dataset_path} "
        f"(base={base_count}, feedback={feedback_count}, final={final_count})"
    )

    original_dataset_path = settings.DATASET_PATH
    settings.DATASET_PATH = str(augmented_dataset_path)

    results: Dict[str, Dict[str, object]] = {}
    try:
        results["lr"] = train_lr(set_production=args.set_production)
        results["nb"] = train_nb(set_production=args.set_production)
        results["svm"] = train_svm(set_production=args.set_production, tune_svm=not args.no_tune_svm)

        if args.skip_lstm:
            results["lstm"] = {"success": True, "skipped": True}
            logger.info("Skipping LSTM retraining by request.")
        else:
            results["lstm"] = train_lstm()
    finally:
        settings.DATASET_PATH = original_dataset_path

    failed = [name for name, output in results.items() if not output.get("success", False)]
    if failed:
        logger.error(f"Retraining finished with failures in: {', '.join(failed)}")
        for name in failed:
            logger.error(f"{name}: {results[name].get('error', 'unknown error')}")
        return 1

    logger.info("Retraining from feedback completed successfully.")
    for name, output in results.items():
        if output.get("skipped"):
            logger.info(f"{name}: skipped")
            continue
        logger.info(f"{name}: success (version={output.get('version', 'n/a')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
