from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class AircraftStateResponse(BaseModel):
    id: int
    icao24: str
    callsign: Optional[str] = None
    latitude: float
    longitude: float
    altitude_m: float
    velocity_ms: float
    heading_deg: float
    vertical_rate_ms: float
    on_ground: bool
    squawk: Optional[str] = None
    received_at: datetime
    source: str

    model_config = {"from_attributes": True, "strict": True, "extra": "forbid"}


class AlertResponse(BaseModel):
    id: int
    icao24: str
    aircraft_state_id: int
    rule_flags: List[str]
    ensemble_score: float
    autoencoder_score: float
    combined_risk_score: float
    reason_text: str
    shap_explanation: Dict[str, Any]
    detected_at: datetime
    is_synthetic: bool
    acknowledged: bool

    model_config = {"from_attributes": True, "strict": True, "extra": "forbid"}


class ModelRunResponse(BaseModel):
    id: int
    run_at: datetime
    model_version: str
    true_positives: int
    false_positives: Optional[int] = None
    true_negatives: Optional[int] = None
    false_negatives: int
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    notes: str

    model_config = {"from_attributes": True, "strict": True, "extra": "forbid"}


class SystemHealthResponse(BaseModel):
    poll_latency_ms: float
    queue_depth: int
    circuit_breaker_state: str
    last_successful_poll: Optional[datetime] = None
    packets_received: int = 0
    packets_processed: int = 0
    packets_dropped: int = 0
    alerts_triggered: int = 0
    active_aircraft: int = 0
    stale_aircraft: int = 0
    last_event_at: Optional[datetime] = None
    source_status: Dict[str, str] = {}

    model_config = {"strict": True, "extra": "forbid"}
