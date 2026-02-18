from fastapi import APIRouter, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from backend.app.services.monitoring_service import monitoring_service

router = APIRouter()

@router.get("/metrics")
async def get_metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@router.get("/dashboard")
async def get_dashboard():
    """Return real-time monitoring data."""
    return monitoring_service.get_dashboard_data()