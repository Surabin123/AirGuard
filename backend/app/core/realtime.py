"""Runtime state and event fan-out for the live AirGuard console.

This module intentionally keeps only the current operational state in memory;
the database remains the source of truth for historical telemetry and alerts.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class LiveAircraftRegistry:
    def __init__(self) -> None:
        self._states: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def upsert(
        self,
        record: Dict[str, Any],
        risk_score: float = 0.0,
        rule_flags: Optional[list[str]] = None,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        icao24 = record["icao24"].lower()
        now = utc_now()
        async with self._lock:
            previous = self._states.get(icao24)
            received_at = record.get("received_at") or now
            state = {
                "icao24": icao24,
                "callsign": record.get("callsign"),
                "latitude": record["latitude"],
                "longitude": record["longitude"],
                "altitude_m": record.get("altitude_m", 0.0),
                "velocity_ms": record.get("velocity_ms", 0.0),
                "heading_deg": record.get("heading_deg", 0.0),
                "vertical_rate_ms": record.get("vertical_rate_ms", 0.0),
                "on_ground": record.get("on_ground", False),
                "squawk": record.get("squawk"),
                "spi": record.get("spi"),
                "planned_route": record.get("planned_route"),
                "route_deviation_nm": record.get("route_deviation_nm"),
                "emergency_code": record.get("emergency_code"),
                "rapid_descent": record.get("rapid_descent", False),
                "events": (previous or {}).get("events", []),
                "received_at": received_at,
                "source": source or record.get("source", "unknown"),
                "risk_score": round(float(risk_score), 4),
                "trust_score": round(max(0.0, 1.0 - float(risk_score)) * 100, 1),
                "rule_flags": rule_flags or [],
                "last_seen_at": now,
                "status": "live",
                "update_count": (previous or {}).get("update_count", 0) + 1,
            }
            sources = set((previous or {}).get("sources", []))
            sources.add(source or record.get("source", "unknown"))
            state["sources"] = sorted(sources)
            state["sensor_consensus"] = round(min(1.0, len(sources) / 3.0), 2)
            previous_callsign = (previous or {}).get("callsign")
            state["identity_conflict"] = bool(
                previous_callsign
                and state["callsign"]
                and previous_callsign != state["callsign"]
            )
            if previous:
                state["previous_latitude"] = previous["latitude"]
                state["previous_longitude"] = previous["longitude"]
            self._states[icao24] = state
            return dict(state)

    async def append_event(self, icao24: str, event: Dict[str, Any]) -> Dict[str, Any]:
        async with self._lock:
            state = self._states.get(icao24.lower())
            if not state:
                return event
            events = list(state.get("events", []))
            events.append(event)
            state["events"] = events[-100:]
            return dict(event)

    async def mark_stale(
        self, stale_after_seconds: float = 45.0, lost_after_seconds: float = 120.0
    ) -> list[Dict[str, Any]]:
        now = utc_now()
        changed: list[Dict[str, Any]] = []
        async with self._lock:
            for state in self._states.values():
                age = (now - state["last_seen_at"]).total_seconds()
                new_status = (
                    "lost"
                    if age >= lost_after_seconds
                    else "stale" if age >= stale_after_seconds else "live"
                )
                if state["status"] != new_status:
                    state["status"] = new_status
                    state["age_seconds"] = round(age, 1)
                    changed.append(dict(state))
        return changed

    async def snapshot(self) -> list[Dict[str, Any]]:
        async with self._lock:
            return [dict(item) for item in self._states.values()]

    async def get(self, icao24: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            item = self._states.get(icao24.lower())
            return dict(item) if item else None


class RuntimeMetrics:
    def __init__(self) -> None:
        self.started_at = utc_now()
        self.packets_received = 0
        self.packets_processed = 0
        self.packets_dropped = 0
        self.alerts_triggered = 0
        self.last_event_at: Optional[datetime] = None
        self.last_processing_ms = 0.0
        self.last_poll_latency_ms = 0.0
        self.source_status: Dict[str, str] = {
            "opensky": "starting",
            "local_sdr": "not_configured",
        }


live_registry = LiveAircraftRegistry()
runtime_metrics = RuntimeMetrics()
incidents: Dict[str, Dict[str, Any]] = {}
