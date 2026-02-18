from fastapi import APIRouter, HTTPException, Response

from ....services.monitoring_service import monitoring_service
from ....services.feedback_service import feedback_service

try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
except ImportError:
    generate_latest = None
    CONTENT_TYPE_LATEST = "text/plain; charset=utf-8"

router = APIRouter()

@router.get("/metrics")
async def get_metrics():
    if generate_latest is None:
        raise HTTPException(
            status_code=503,
            detail="prometheus_client is not installed; install it to enable /metrics"
        )
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@router.get("/dashboard")
async def get_dashboard():
    """Return real-time monitoring data."""
    dashboard_data = monitoring_service.get_dashboard_data()
    dashboard_data["feedback_stats"] = feedback_service.get_stats()
    return dashboard_data
