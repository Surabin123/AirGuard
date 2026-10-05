from fastapi import APIRouter, Depends, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional
from datetime import datetime, timezone
from io import BytesIO
from fastapi.responses import StreamingResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

from app.core.database import get_db
from app.core.limiter import limiter
from app.models import AircraftState, Alert, Geofence, Incident, ModelRun
from app.api.schemas import AircraftStateResponse, AlertResponse, ModelRunResponse, SystemHealthResponse
from app.core.realtime import incidents, live_registry, runtime_metrics

router = APIRouter()


class GeofencePayload(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    radius_km: float = Field(gt=0, le=500)
    enabled: bool = True


@router.get("/geofences")
@limiter.limit("30/minute")
async def list_geofences(request: Request, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Geofence).order_by(Geofence.id.asc()))
    return result.scalars().all()


@router.post("/geofences")
@limiter.limit("20/minute")
async def create_geofence(request: Request, payload: GeofencePayload, db: AsyncSession = Depends(get_db)):
    geofence = Geofence(**payload.model_dump(), created_at=datetime.now(timezone.utc))
    db.add(geofence)
    await db.commit()
    await db.refresh(geofence)
    return geofence


@router.patch("/geofences/{geofence_id}")
@limiter.limit("30/minute")
async def update_geofence(request: Request, geofence_id: int, payload: GeofencePayload, db: AsyncSession = Depends(get_db)):
    geofence = await db.get(Geofence, geofence_id)
    if not geofence:
        raise HTTPException(status_code=404, detail="Geofence not found")
    for key, value in payload.model_dump().items():
        setattr(geofence, key, value)
    await db.commit()
    await db.refresh(geofence)
    return geofence


@router.delete("/geofences/{geofence_id}")
@limiter.limit("20/minute")
async def delete_geofence(request: Request, geofence_id: int, db: AsyncSession = Depends(get_db)):
    geofence = await db.get(Geofence, geofence_id)
    if not geofence:
        raise HTTPException(status_code=404, detail="Geofence not found")
    await db.delete(geofence)
    await db.commit()
    return {"deleted": True, "id": geofence_id}

# Keep track of service startup time for session reporting
START_TIME = datetime.now(timezone.utc)

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(jsonable_encoder(message))
            except Exception:
                pass

manager = ConnectionManager()

# Most recent values sampled from the active ingestion service.
SYSTEM_STATS = {
    "poll_latency_ms": 0.0,
    "queue_depth": 0,
    "circuit_breaker_state": "UNKNOWN",
    "last_successful_poll": None
}

@router.get("/aircraft", response_model=List[AircraftStateResponse])
@limiter.limit("50/minute")
async def get_aircraft(
    request: Request,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    """Retrieve recent aircraft states, paginated."""
    result = await db.execute(
        select(AircraftState)
        .order_by(AircraftState.received_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return result.scalars().all()


@router.get("/aircraft/history", response_model=List[AircraftStateResponse])
@limiter.limit("30/minute")
async def get_all_aircraft_history(
    request: Request,
    start: datetime = Query(...),
    end: datetime = Query(...),
    db: AsyncSession = Depends(get_db)
):
    """Retrieve historical states for all aircraft within a time range."""
    result = await db.execute(
        select(AircraftState)
        .where(AircraftState.received_at >= start)
        .where(AircraftState.received_at <= end)
        .order_by(AircraftState.received_at.asc())
    )
    return result.scalars().all()


@router.get("/aircraft/{icao24}/history", response_model=List[AircraftStateResponse])
@limiter.limit("50/minute")
async def get_aircraft_history(
    request: Request,
    icao24: str,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    """Retrieve historical reports for a specific aircraft address (ICAO 24-bit)."""
    result = await db.execute(
        select(AircraftState)
        .where(AircraftState.icao24 == icao24.lower())
        .order_by(AircraftState.received_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return result.scalars().all()


@router.get("/aircraft/{icao24}/replay", response_model=List[AircraftStateResponse])
@limiter.limit("20/minute")
async def replay_aircraft_history(
    request: Request,
    icao24: str,
    start: Optional[datetime] = Query(default=None),
    end: Optional[datetime] = Query(default=None),
    limit: int = Query(default=1000, ge=1, le=10000),
    db: AsyncSession = Depends(get_db),
):
    """Return only recorded telemetry in event-time order."""
    query = select(AircraftState).where(AircraftState.icao24 == icao24.lower())
    if start:
        query = query.where(AircraftState.received_at >= start)
    if end:
        query = query.where(AircraftState.received_at <= end)
    result = await db.execute(query.order_by(AircraftState.received_at.asc()).limit(limit))
    return result.scalars().all()


@router.get("/alerts", response_model=List[AlertResponse])
@limiter.limit("30/minute")
async def get_alerts(
    request: Request,
    acknowledged: Optional[bool] = Query(default=None),
    icao24: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    """Retrieve security anomaly alerts, filterable by acknowledged state and ICAO code."""
    query = select(Alert)
    
    if acknowledged is not None:
        query = query.where(Alert.acknowledged == acknowledged)
    if icao24 is not None:
        query = query.where(Alert.icao24 == icao24.lower())
        
    result = await db.execute(
        query.order_by(Alert.detected_at.desc()).limit(limit).offset(offset)
    )
    return result.scalars().all()


@router.post("/alerts/{id}/acknowledge", response_model=AlertResponse)
@limiter.limit("20/minute")
async def acknowledge_alert(
    request: Request,
    id: int,
    db: AsyncSession = Depends(get_db)
):
    """Acknowledge a specific security alert."""
    result = await db.execute(
        select(Alert).where(Alert.id == id)
    )
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
        
    alert.acknowledged = True
    await db.commit()
    await db.refresh(alert)
    return alert


@router.get("/model-runs", response_model=List[ModelRunResponse])
@limiter.limit("30/minute")
async def get_model_runs(
    request: Request,
    limit: int = Query(default=50, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """Retrieve historical machine learning model evaluation runs."""
    result = await db.execute(
        select(ModelRun).order_by(ModelRun.run_at.desc()).limit(limit)
    )
    return result.scalars().all()


@router.get("/system-health", response_model=SystemHealthResponse)
@limiter.limit("60/minute")
async def get_system_health(request: Request):
    """Get system ingestion statistics, queue depths, and circuit-breaker status."""
    live_states = await live_registry.snapshot()
    stale_count = sum(1 for state in live_states if state.get("status") in {"stale", "lost"})
    return SystemHealthResponse(
        poll_latency_ms=SYSTEM_STATS["poll_latency_ms"],
        queue_depth=SYSTEM_STATS["queue_depth"],
        circuit_breaker_state=SYSTEM_STATS["circuit_breaker_state"],
        last_successful_poll=SYSTEM_STATS["last_successful_poll"],
        packets_received=runtime_metrics.packets_received,
        packets_processed=runtime_metrics.packets_processed,
        packets_dropped=runtime_metrics.packets_dropped,
        alerts_triggered=runtime_metrics.alerts_triggered,
        active_aircraft=len(live_states),
        stale_aircraft=stale_count,
        last_event_at=runtime_metrics.last_event_at,
        source_status=dict(runtime_metrics.source_status)
    )


@router.get("/live-aircraft")
@limiter.limit("60/minute")
async def get_live_aircraft(request: Request):
    """Return the current operational aircraft registry, including freshness."""
    return await live_registry.snapshot()


@router.get("/live-aircraft/{icao24}")
@limiter.limit("60/minute")
async def get_live_aircraft_detail(request: Request, icao24: str):
    state = await live_registry.get(icao24)
    if not state:
        raise HTTPException(status_code=404, detail="Aircraft is not currently tracked")
    return state


@router.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint to subscribe to real-time ADS-B states and security alert broadcasts."""
    await manager.connect(websocket)
    try:
        while True:
            # Block and wait for messages (primarily to detect client disconnection)
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


class TelemetryPayload(BaseModel):
    """Normalized telemetry contract for live, authorized data sources."""
    icao24: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    altitude_m: float = 0.0
    velocity_ms: float = 0.0
    heading_deg: float = 0.0
    vertical_rate_ms: float = 0.0
    on_ground: bool = False
    squawk: Optional[str] = Field(default=None, pattern=r"^\d{4}$")
    callsign: Optional[str] = None
    received_at: Optional[datetime] = None
    source: str = Field(default="external", min_length=1, max_length=40)
    receiver_id: Optional[str] = Field(default=None, max_length=30)
    planned_route: Optional[List["RouteWaypoint"]] = Field(default=None, min_length=2)
    route_tolerance_nm: float = Field(default=10.0, gt=0, le=100)


class RouteWaypoint(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


TelemetryPayload.model_rebuild()

@router.post("/inject")
@limiter.limit("60/minute")
async def inject_anomaly(request: Request):
    """Synthetic injection is disabled; live telemetry must come from an authorized source."""
    raise HTTPException(status_code=410, detail="Synthetic anomaly injection is disabled. Submit real normalized telemetry to /api/v1/ingest.")


@router.post("/ingest")
@limiter.limit("600/minute")
async def ingest_telemetry(request: Request, payload: TelemetryPayload):
    """Ingest one normalized live state from an SDR bridge or another feed."""
    record = payload.model_dump()
    record["icao24"] = record["icao24"].strip().lower()
    record["received_at"] = record["received_at"] or datetime.now(timezone.utc)
    record["metadata"] = {"is_known_entity": False, "known_entity_label": None}
    record["source"] = payload.source if not payload.receiver_id else f"{payload.source}:{payload.receiver_id}"
    runtime_metrics.packets_received += 1
    runtime_metrics.source_status[payload.source] = "healthy"
    await ingestion_queue.put(record)
    return {"status": "accepted", "icao24": record["icao24"], "source": payload.source}


@router.get("/incidents")
@limiter.limit("60/minute")
async def get_incidents(
    request: Request,
    limit: int = Query(default=100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Incident).order_by(Incident.created_at.desc()).limit(limit))
    return result.scalars().all()


class IncidentUpdatePayload(BaseModel):
    status: str
    comment: Optional[str] = None


@router.get("/capabilities")
@limiter.limit("60/minute")
async def get_live_capabilities(request: Request):
    return {
        "live_state_fields": ["position", "altitude", "velocity", "vertical_rate", "squawk", "source"],
        "emergency_squawk_codes": ["7500", "7600", "7700"],
        "rapid_descent_detection": True,
        "filed_route_deviation": {
            "available": True,
            "requires_input": "planned_route on POST /ingest from a filed-flight-plan provider",
            "provided_by_opensky_state_vectors": False,
        },
        "historical_replay": "recorded_database_telemetry_only",
        "incident_timeline": "postgresql_persisted",
        "diversion_airports": {
            "available": False,
            "reason": "No airport, runway availability, or operational suitability provider is configured.",
        },
    }


@router.post("/incidents/{incident_id}")
@limiter.limit("60/minute")
async def update_incident(
    request: Request,
    incident_id: str,
    payload: IncidentUpdatePayload,
    db: AsyncSession = Depends(get_db),
):
    if payload.status not in {"new", "investigating", "acknowledged", "resolved", "archived"}:
        raise HTTPException(status_code=400, detail="Unsupported incident status")
    result = await db.execute(select(Incident).where(Incident.id == incident_id))
    incident = result.scalar_one_or_none()
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    at = datetime.now(timezone.utc)
    incident.status = payload.status
    incident.updated_at = at
    timeline = list(incident.timeline or [])
    timeline.append({
        "at": at.isoformat(),
        "type": "operator_status_change",
        "signals": [f"status:{payload.status}"],
    })
    incident.timeline = timeline
    if payload.comment:
        comments = list(incident.comments or [])
        comments.append({"at": at.isoformat(), "text": payload.comment})
        incident.comments = comments
        incident.timeline = timeline + [{"at": at.isoformat(), "type": "operator_comment", "text": payload.comment, "signals": []}]
    await db.commit()
    await db.refresh(incident)
    incident_payload = {
        "id": incident.id, "icao24": incident.icao24, "status": incident.status,
        "created_at": incident.created_at.isoformat(), "updated_at": incident.updated_at.isoformat(),
        "risk_score": incident.risk_score, "reason": incident.reason,
        "rule_flags": incident.rule_flags, "comments": incident.comments,
        "timeline": incident.timeline, "source": incident.source,
    }
    incidents[incident.id] = incident_payload
    await manager.broadcast({"event": "INCIDENT_UPDATED", "payload": incident_payload})
    return incident_payload


@router.get("/reports/session")
@limiter.limit("5/minute")
async def generate_session_report(
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """Generate a high-fidelity PDF session report including stats, latest model run, and top 5 risk alerts with SHAP explanations."""
    # 1. Gather Session Metrics
    duration = datetime.now(timezone.utc) - START_TIME
    hours, remainder = divmod(int(duration.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    duration_str = f"{hours}h {minutes}m {seconds}s"
    
    # Tracked aircraft count
    aircraft_count_result = await db.execute(
        select(AircraftState.icao24).distinct()
    )
    tracked_aircraft_count = len(aircraft_count_result.scalars().all())
    
    # Alerts count
    alerts_result = await db.execute(
        select(Alert)
    )
    all_alerts = alerts_result.scalars().all()
    total_alerts = len(all_alerts)
    
    # Alerts by type
    alerts_by_type = {}
    for a in all_alerts:
        for flag in a.rule_flags:
            alerts_by_type[flag] = alerts_by_type.get(flag, 0) + 1
            
    # Latest Model Run
    model_run_result = await db.execute(
        select(ModelRun).order_by(ModelRun.run_at.desc()).limit(1)
    )
    latest_run = model_run_result.scalar_one_or_none()
    
    # Top 5 highest-risk alerts
    top_alerts_result = await db.execute(
        select(Alert).order_by(Alert.combined_risk_score.desc()).limit(5)
    )
    top_alerts = top_alerts_result.scalars().all()

    # 2. Build PDF report via ReportLab
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0ea5e9'),
        spaceAfter=12
    )
    
    h2_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#0f172a'),
        spaceBefore=12,
        spaceAfter=6
    )
    
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155')
    )

    bold_body_style = ParagraphStyle(
        'ReportBodyBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    story = []
    
    # Header Title
    story.append(Paragraph("AIRGUARD // SESSION AUDIT REPORT", title_style))
    story.append(Paragraph(f"Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC", body_style))
    story.append(Spacer(1, 10))
    
    # Session Details Table
    story.append(Paragraph("1. Session Metrics Summary", h2_style))
    stats_data = [
        [Paragraph("Session Duration", bold_body_style), Paragraph(duration_str, body_style)],
        [Paragraph("Total Tracked Aircraft", bold_body_style), Paragraph(str(tracked_aircraft_count), body_style)],
        [Paragraph("Total Security Alerts", bold_body_style), Paragraph(str(total_alerts), body_style)]
    ]
    t1 = Table(stats_data, colWidths=[200, 300])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#94a3b8')),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t1)
    story.append(Spacer(1, 12))
    
    # Alerts By Type Table
    story.append(Paragraph("2. Alerts Distribution by Rule Type", h2_style))
    type_data = [[Paragraph("Rule Type", bold_body_style), Paragraph("Count", bold_body_style)]]
    if len(alerts_by_type) == 0:
        type_data.append([Paragraph("No alerts logged", body_style), Paragraph("0", body_style)])
    for k, v in alerts_by_type.items():
        type_data.append([Paragraph(k, body_style), Paragraph(str(v), body_style)])
    t2 = Table(type_data, colWidths=[200, 300])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#94a3b8')),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t2)
    story.append(Spacer(1, 12))
    
    # Model Run Precision/Recall Table
    story.append(Paragraph("3. Active Classifier Model Verification", h2_style))
    if latest_run:
        mr_data = [
            [Paragraph("Attribute", bold_body_style), Paragraph("Value", bold_body_style)],
            [Paragraph("Model Version", body_style), Paragraph(latest_run.model_version, body_style)],
            [Paragraph("Precision", body_style), Paragraph(f"{latest_run.precision:.4f}" if latest_run.precision is not None else "Unavailable (no labeled normal samples)", body_style)],
            [Paragraph("Recall", body_style), Paragraph(f"{latest_run.recall:.4f}" if latest_run.recall is not None else "Unavailable (no labeled anomaly samples)", body_style)],
            [Paragraph("F1 Score", body_style), Paragraph(f"{latest_run.f1:.4f}" if latest_run.f1 is not None else "Unavailable (requires both labeled classes)", body_style)],
            [Paragraph("Notes / Training Set size", body_style), Paragraph(latest_run.notes or "N/A", body_style)],
        ]
    else:
        mr_data = [
            [Paragraph("Status", bold_body_style), Paragraph("No model runs recorded yet", body_style)]
        ]
    t3 = Table(mr_data, colWidths=[200, 300])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#94a3b8')),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t3)
    story.append(Spacer(1, 12))
    
    # Top 5 Highest Risk Alerts
    story.append(Paragraph("4. Top 5 Highest Risk Anomalies & SHAP Explanations", h2_style))
    if len(top_alerts) == 0:
        story.append(Paragraph("No security alerts logged during session.", body_style))
    else:
        for idx, alert in enumerate(top_alerts, 1):
            story.append(Paragraph(f"<b>Anomaly {idx}: {alert.callsign} ({alert.icao24.upper()})</b>", bold_body_style))
            story.append(Paragraph(f"Combined Risk Score: <b>{alert.combined_risk_score:.4f}</b>", body_style))
            story.append(Paragraph(f"Reason: <i>{alert.reason_text}</i>", body_style))
            
            # Map SHAP values if present
            shap_text = "N/A"
            if isinstance(alert.shap_explanation, dict) and "shap" in alert.shap_explanation:
                shap_list = alert.shap_explanation["shap"]
                if isinstance(shap_list, dict):
                    shap_items = [f"{k}: {v:.4f}" for k, v in shap_list.items() if v > 0]
                    shap_text = ", ".join(shap_items) if len(shap_items) > 0 else "Low feature contributions"
            
            story.append(Paragraph(f"SHAP Explanations: {shap_text}", body_style))
            story.append(Spacer(1, 6))

    doc.build(story)
    buffer.seek(0)
    
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment;filename=airguard_session_report.pdf"}
    )


from app.core.rule_config import active_rule_config
from app.detection.rules import (
    check_impossible_climb_rate,
    check_altitude_velocity_mismatch,
    check_position_jump,
    check_duplicate_icao
)

class ConfigUpdatePayload(BaseModel):
    max_implied_speed_kmh: float
    duplicate_icao_dist_km: float
    max_vertical_rate_ms: float
    max_ground_altitude_m: float
    max_ground_speed_ms: float
    min_flight_speed_ms: float

    model_config = {
        "strict": True,
        "extra": "forbid"
    }

@router.get("/config")
@limiter.limit("30/minute")
async def get_current_config(request: Request):
    """Retrieve the current active thresholds config."""
    return active_rule_config


@router.post("/config")
@limiter.limit("10/minute")
async def update_thresholds_config(
    request: Request,
    payload: ConfigUpdatePayload
):
    """Update active telemetry rules check thresholds."""
    active_rule_config.max_implied_speed_kmh = payload.max_implied_speed_kmh
    active_rule_config.duplicate_icao_dist_km = payload.duplicate_icao_dist_km
    active_rule_config.max_vertical_rate_ms = payload.max_vertical_rate_ms
    active_rule_config.max_ground_altitude_m = payload.max_ground_altitude_m
    active_rule_config.max_ground_speed_ms = payload.max_ground_speed_ms
    active_rule_config.min_flight_speed_ms = payload.min_flight_speed_ms
    return active_rule_config


@router.post("/model-runs/replay", response_model=ModelRunResponse)
@limiter.limit("5/minute")
async def replay_session_validation(
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """Replay stored telemetry; calculate only metrics supported by recorded labels."""
    # 1. Fetch historical states and alerts
    states_result = await db.execute(
        select(AircraftState).order_by(AircraftState.icao24, AircraftState.received_at.asc())
    )
    all_states = states_result.scalars().all()

    alerts_result = await db.execute(
        select(Alert)
    )
    all_alerts = alerts_result.scalars().all()

    # Map state_id to whether it was a ground truth synthetic anomaly
    synthetic_state_ids = {a.aircraft_state_id for a in all_alerts if a.is_synthetic}

    TP, FN = 0, 0

    # We evaluate sequentially grouped by ICAO to handle history-based checks (jump, duplicate)
    icao_histories = {}

    for state in all_states:
        icao = state.icao24
        history = icao_histories.get(icao, [])

        # Evaluate rules
        rule_climb, _, _ = check_impossible_climb_rate(
            state.vertical_rate_ms, active_rule_config
        )
        rule_alt_vel, _, _ = check_altitude_velocity_mismatch(
            state.altitude_m, state.velocity_ms, state.on_ground, active_rule_config
        )

        rule_jump = False
        rule_dup = False
        if len(history) > 0:
            prev = history[-1]
            rule_jump, _, _ = check_position_jump(
                current_lat=state.latitude, current_lon=state.longitude, current_time=state.received_at,
                prev_lat=prev.latitude, prev_lon=prev.longitude, prev_time=prev.received_at,
                config=active_rule_config
            )
            rule_dup, _, _ = check_duplicate_icao(
                lat_a=state.latitude, lon_a=state.longitude, time_a=state.received_at,
                lat_b=prev.latitude, lon_b=prev.longitude, time_b=prev.received_at,
                config=active_rule_config
            )

        predicted_anomaly = rule_climb or rule_alt_vel or rule_jump or rule_dup
        # Unlabeled live traffic is not ground truth negative data. Only explicitly
        # labeled synthetic anomaly observations can contribute to recall here.
        if state.id in synthetic_state_ids:
            if predicted_anomaly:
                TP += 1
            else:
                FN += 1

        history.append(state)
        icao_histories[icao] = history

    # Calculate metrics
    recall = TP / (TP + FN) if (TP + FN) > 0 else None

    # 2. Write new ModelRun row
    db_run = ModelRun(
        run_at=datetime.now(timezone.utc),
        model_version="Replay-Config",
        true_positives=TP,
        false_positives=None,
        true_negatives=None,
        false_negatives=FN,
        precision=None,
        recall=recall,
        f1=None,
        notes=(f"Positive-only labeled replay: {TP + FN} labeled anomalies; recall is measurable. "
               "Precision/F1 and false-positive/true-negative counts require labeled normal observations. "
               f"Thresholds: climb={active_rule_config.max_vertical_rate_ms}m/s, speed={active_rule_config.max_implied_speed_kmh}km/h")
    )
    db.add(db_run)
    await db.commit()
    await db.refresh(db_run)

    return db_run
