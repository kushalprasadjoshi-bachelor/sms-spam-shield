import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from ..core.config import settings
from ..core.logger import logger
from ..schemas.feedback import FeedbackRecord, FeedbackRequest


class FeedbackService:
    def __init__(self, store_path: str):
        self._lock = threading.Lock()
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_store()

    def _ensure_store(self) -> None:
        if self.store_path.exists():
            return
        self._write_records([])

    def _read_records(self) -> List[Dict[str, Any]]:
        try:
            with self.store_path.open("r", encoding="utf-8") as file_obj:
                payload = json.load(file_obj)
            if isinstance(payload, list):
                return payload
            return payload.get("feedback", []) if isinstance(payload, dict) else []
        except Exception as exc:
            logger.warning(f"Failed to read feedback store: {exc}")
            return []

    def _write_records(self, records: List[Dict[str, Any]]) -> None:
        with self.store_path.open("w", encoding="utf-8") as file_obj:
            json.dump({"feedback": records}, file_obj, indent=2)

    @staticmethod
    def _to_feedback_record(data: Dict[str, Any]) -> FeedbackRecord:
        if hasattr(FeedbackRecord, "model_validate"):
            return FeedbackRecord.model_validate(data)
        return FeedbackRecord.parse_obj(data)

    @staticmethod
    def _normalize_models(models: List[str]) -> List[str]:
        normalized = []
        seen = set()
        for model in models:
            value = str(model).strip().lower()
            if not value or value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        return normalized

    def upsert_feedback(self, request: FeedbackRequest) -> FeedbackRecord:
        with self._lock:
            records = self._read_records()
            now = datetime.now(timezone.utc)

            predicted_label = (request.predicted_label or "").strip().lower() or None
            corrected_label = request.corrected_label.strip().lower()
            sms = (request.sms or "").strip()
            selected_models = self._normalize_models(request.selected_models)
            is_correct = predicted_label == corrected_label if predicted_label else None

            existing = next(
                (item for item in records if str(item.get("prediction_id", "")) == request.prediction_id),
                None
            )

            if existing:
                existing.update({
                    "corrected_label": corrected_label,
                    "predicted_label": predicted_label,
                    "sms": sms,
                    "selected_models": selected_models,
                    "is_correct": is_correct,
                    "updated_at": now.isoformat()
                })
                if not existing.get("created_at"):
                    existing["created_at"] = now.isoformat()
                record_data = existing
            else:
                record_data = {
                    "prediction_id": request.prediction_id,
                    "corrected_label": corrected_label,
                    "predicted_label": predicted_label,
                    "sms": sms,
                    "selected_models": selected_models,
                    "is_correct": is_correct,
                    "created_at": now.isoformat(),
                    "updated_at": now.isoformat()
                }
                records.append(record_data)

            self._write_records(records)
            return self._to_feedback_record(record_data)

    def list_feedback(self) -> List[FeedbackRecord]:
        with self._lock:
            return [self._to_feedback_record(item) for item in self._read_records()]

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            records = self._read_records()

        total_feedback = len(records)
        total_corrections = 0
        total_confirmed_correct = 0
        corrections_by_model: Dict[str, int] = {}
        corrections_by_label: Dict[str, int] = {}

        for record in records:
            is_correct = record.get("is_correct")
            corrected_label = str(record.get("corrected_label", "")).strip().lower()
            selected_models = record.get("selected_models") or []

            if is_correct is True:
                total_confirmed_correct += 1
            else:
                total_corrections += 1
                if corrected_label:
                    corrections_by_label[corrected_label] = corrections_by_label.get(corrected_label, 0) + 1

                model_keys = selected_models if isinstance(selected_models, list) and selected_models else ["ensemble"]
                for model_key in model_keys:
                    model_name = str(model_key).strip().lower()
                    if not model_name:
                        continue
                    corrections_by_model[model_name] = corrections_by_model.get(model_name, 0) + 1

        return {
            "total_feedback": total_feedback,
            "total_corrections": total_corrections,
            "total_confirmed_correct": total_confirmed_correct,
            "corrections_by_model": corrections_by_model,
            "corrections_by_label": corrections_by_label
        }


feedback_service = FeedbackService(settings.FEEDBACK_STORE_PATH)
