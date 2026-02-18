import time
from collections import deque
from typing import Dict, Optional
import threading


class MonitoringService:
    def __init__(self, maxlen=1000):
        self.lock = threading.Lock()
        self.recent = deque(maxlen=maxlen)

    def record_prediction(self, model: str, category: str, confidence: float, correct: Optional[bool] = None):
        with self.lock:
            self.recent.append({
                'timestamp': time.time(),
                'model': model,
                'category': category,
                'confidence': confidence,
                'correct': correct
            })

    def get_dashboard_data(self) -> Dict:
        with self.lock:
            total = len(self.recent)
            if total == 0:
                return {'total_predictions': 0}

            model_counts = {}
            cat_counts = {}
            confidences = {}
            correct_count = 0

            for p in self.recent:
                model_counts[p['model']] = model_counts.get(p['model'], 0) + 1
                cat_counts[p['category']] = cat_counts.get(p['category'], 0) + 1
                confidences.setdefault(p['model'], []).append(p['confidence'])
                if p['correct']:
                    correct_count += 1

            avg_conf = {m: sum(v)/len(v) for m, v in confidences.items()}
            accuracy = correct_count / total if correct_count > 0 else None

            now = time.time()
            throughput = []
            for i in range(10):
                start = now - (i+1)*6
                end = now - i*6
                cnt = sum(1 for p in self.recent if start <= p['timestamp'] < end)
                throughput.append({'time': end, 'count': cnt})

            return {
                'total_predictions': total,
                'model_counts': model_counts,
                'category_distribution': cat_counts,
                'average_confidence': avg_conf,
                'accuracy': accuracy,
                'throughput': throughput[::-1]
            }


monitoring_service = MonitoringService()