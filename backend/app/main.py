################################ MAIN APPLICATION ENTRY POINT ####################################
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pathlib import Path

from .core.config import settings
from .core.logger import logger
from .api.v1.endpoints import predict, health, compare, monitoring, models
from .services.model_manager import model_manager

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="SMS Spam Shield: Multi-Category XAI Spam Detector",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

project_root = Path(__file__).resolve().parents[2]

# Mount static files if they exist
frontend_static_path = project_root / "frontend" / "static"
logger.info(f"Static path: {frontend_static_path}")
logger.info(f"Static path exists: {frontend_static_path.exists()}")
if frontend_static_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_static_path)), name="static")

# Templates
templates_path = project_root / "frontend" / "templates"
templates = Jinja2Templates(directory=str(templates_path)) if templates_path.exists() else None

# Include routers
app.include_router(predict.router, prefix=settings.API_V1_PREFIX)
app.include_router(health.router, prefix=settings.API_V1_PREFIX)
app.include_router(compare.router, prefix=settings.API_V1_PREFIX)
app.include_router(monitoring.router, prefix=settings.API_V1_PREFIX)
app.include_router(models.router, prefix=settings.API_V1_PREFIX)

@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    """Serve the main frontend page"""
    if templates:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "project_name": settings.PROJECT_NAME}
        )
    return HTMLResponse("""
        <html>
            <head><title>SMS Spam Shield</title></head>
            <body>
                <h1>SMS Spam Shield Backend is running!</h1>
                <p>Please ensure the frontend files are in the correct location.</p>
                <p><a href="/api/docs">API Documentation</a></p>
            </body>
        </html>
    """)


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    
    # Load models on startup
    try:
        logger.info("Loading models...")
        load_results = model_manager.load_all_models()
        
        loaded_count = sum(1 for loaded in load_results.values() if loaded)
        logger.info(f"Loaded {loaded_count}/{len(load_results)} models")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down SMS Spam Shield...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
