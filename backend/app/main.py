import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pythonjsonlogger import jsonlogger
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy import text

from app.core.config import settings
from app.core.database import async_session_maker, get_db
from app.core.limiter import limiter
from app.core.queue import ingestion_queue
from app.core.realtime import live_registry, runtime_metrics
from app.api.v1.endpoints import SYSTEM_STATS, manager, router as api_router

# Import Ingestion & Detection pipeline modules
from app.ingestion.service import OpenSkyIngestionService
from app.detection.service import DetectionService
from app.detection.ensemble import TrustScoringEnsemble
from app.detection.autoencoder import UnsupervisedAutoencoder

# Configure JSON Logging
logger = logging.getLogger()
log_handler = logging.StreamHandler(sys.stdout)
formatter = jsonlogger.JsonFormatter(
    fmt="%(asctime)s %(levelname)s %(name)s %(message)s"
)
log_handler.setFormatter(formatter)
logger.addHandler(log_handler)
logger.setLevel(settings.LOG_LEVEL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """LIFESPAN: Initializes models, services, and runs background ingestion/detection loops."""
    logger.info("Initializing AirGuard backend models and background loops...")

    # 1. Initialize ML/DL models
    ensemble = TrustScoringEnsemble()
    autoencoder = UnsupervisedAutoencoder()

    # 2. Instantiate pipeline services
    ingestion_service = OpenSkyIngestionService(
        queue=ingestion_queue,
        db_session_maker=async_session_maker,
        poll_interval_seconds=8.0,
    )
    detection_service = DetectionService(
        queue=ingestion_queue,
        db_session_maker=async_session_maker,
        ensemble_model=ensemble,
        autoencoder_model=autoencoder,
    )

    # 3. Spin up concurrent background loop tasks
    ingestion_task = asyncio.create_task(ingestion_service.start_polling_loop())
    detection_task = asyncio.create_task(detection_service.start_detection_loop())

    # 4. Background task syncing ingestion statistics to API health endpoint
    async def sync_stats_loop():
        while True:
            try:
                SYSTEM_STATS["poll_latency_ms"] = runtime_metrics.last_poll_latency_ms
                SYSTEM_STATS["queue_depth"] = ingestion_queue.qsize()
                SYSTEM_STATS["circuit_breaker_state"] = ingestion_service.breaker_state
                SYSTEM_STATS["last_successful_poll"] = (
                    ingestion_service.last_successful_poll
                )
                runtime_metrics.source_status["opensky"] = (
                    "healthy"
                    if ingestion_service.last_successful_poll
                    and ingestion_service.consecutive_failures == 0
                    else (
                        "degraded"
                        if ingestion_service.consecutive_failures
                        else "starting"
                    )
                )
                for state in await live_registry.mark_stale():
                    await manager.broadcast(
                        {
                            "event": "AIRCRAFT_STATUS_CHANGED",
                            "payload": state,
                        }
                    )
            except Exception:
                pass
            await asyncio.sleep(2)

    stats_task = asyncio.create_task(sync_stats_loop())

    logger.info("AirGuard background tasks started successfully.")
    yield

    # 5. Shutdown sequence - Cancel active background tasks
    logger.info("Shutting down AirGuard background tasks...")
    ingestion_task.cancel()
    detection_task.cancel()
    stats_task.cancel()

    # Wait for tasks to close gracefully
    await asyncio.gather(
        ingestion_task, detection_task, stats_task, return_exceptions=True
    )
    logger.info("Shutdown sequence completed.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# Include Router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Set limiter on app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def health_check(request: Request, db=Depends(get_db)):
    try:
        # Check DB connectivity
        await db.execute(text("SELECT 1"))
        return {
            "status": "healthy",
            "database": "connected",
            "service": settings.PROJECT_NAME,
        }
    except Exception as e:
        logger.error(f"Health check failed database check: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "database": "disconnected",
                "reason": str(e),
            },
        )
