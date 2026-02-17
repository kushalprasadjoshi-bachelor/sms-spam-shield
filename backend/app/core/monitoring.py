from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time
from functools import wraps

prediction_total = Counter('prediction_total', 'Total predictions', ['model'])
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency', ['model'])
model_confidence = Gauge('model_confidence', 'Confidence per model', ['model', 'category'])

def track_prediction(model_name):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            result = await func(*args, **kwargs)
            latency = time.time() - start
            prediction_latency.labels(model=model_name).observe(latency)
            prediction_total.labels(model=model_name).inc()
            return result
        return wrapper
    return decorator