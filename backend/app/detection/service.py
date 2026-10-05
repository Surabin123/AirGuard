import asyncio
import json
import logging
import time
from math import asin, atan2, cos, radians, sin, sqrt
from datetime import datetime, timezone
import numpy as np
from typing import Any, Dict, List

from app.models import AircraftState, Alert, Geofence, Incident
from sqlalchemy import select
from app.detection.rules import (
    RuleConfig,
    check_position_jump,
    check_duplicate_icao,
    check_impossible_climb_rate,
    check_altitude_velocity_mismatch,
)
from app.core.rule_config import active_rule_config
from app.detection.ensemble import FEATURE_NAMES, TrustScoringEnsemble
from app.detection.autoencoder import (
    UnsupervisedAutoencoder,
    check_trilateration_plausibility,
    combine_scores,
)
from app.core.realtime import live_registry, runtime_metrics

logger = logging.getLogger("airguard.detection")

EMERGENCY_SQUAWKS = {
    "7500": "Unlawful interference code (7500)",
    "7600": "Radio communication failure code (7600)",
    "7700": "General emergency code (7700)",
}


def _distance_to_route_nm(
    latitude: float, longitude: float, route: list[dict]
) -> float | None:
    """Great-circle cross-track distance in nautical miles to a filed-route polyline."""
    if len(route) < 2:
        return None

    earth_radius_nm = 3440.065

    def angular_distance(
        lat_a: float, lon_a: float, lat_b: float, lon_b: float
    ) -> float:
        lat_a, lon_a, lat_b, lon_b = map(radians, (lat_a, lon_a, lat_b, lon_b))
        delta_lat, delta_lon = lat_b - lat_a, lon_b - lon_a
        hav = (
            sin(delta_lat / 2) ** 2 + cos(lat_a) * cos(lat_b) * sin(delta_lon / 2) ** 2
        )
        return 2 * atan2(sqrt(max(0.0, hav)), sqrt(max(0.0, 1.0 - hav)))

    def initial_bearing(
        lat_a: float, lon_a: float, lat_b: float, lon_b: float
    ) -> float:
        lat_a, lon_a, lat_b, lon_b = map(radians, (lat_a, lon_a, lat_b, lon_b))
        delta_lon = lon_b - lon_a
        return atan2(
            sin(delta_lon) * cos(lat_b),
            cos(lat_a) * sin(lat_b) - sin(lat_a) * cos(lat_b) * cos(delta_lon),
        )

    closest = float("inf")
    for start, end in zip(route, route[1:]):
        start_lat, start_lon = float(start["latitude"]), float(start["longitude"])
        end_lat, end_lon = float(end["latitude"]), float(end["longitude"])
        segment = angular_distance(start_lat, start_lon, end_lat, end_lon)
        distance_from_start = angular_distance(
            start_lat, start_lon, latitude, longitude
        )
        bearing_to_point = initial_bearing(start_lat, start_lon, latitude, longitude)
        bearing_to_end = initial_bearing(start_lat, start_lon, end_lat, end_lon)
        angle_delta = bearing_to_point - bearing_to_end
        cross_track = asin(
            max(-1.0, min(1.0, sin(distance_from_start) * sin(angle_delta)))
        )
        along_track = atan2(
            sin(distance_from_start) * cos(angle_delta), cos(distance_from_start)
        )
        if along_track < 0:
            distance = angular_distance(start_lat, start_lon, latitude, longitude)
        elif along_track > segment:
            distance = angular_distance(end_lat, end_lon, latitude, longitude)
        else:
            distance = abs(cross_track)
        closest = min(closest, distance * earth_radius_nm)
    return closest


class DetectionService:
    def __init__(
        self,
        queue: asyncio.Queue,
        db_session_maker: Any,
        ensemble_model: TrustScoringEnsemble,
        autoencoder_model: UnsupervisedAutoencoder,
        rule_config: RuleConfig = active_rule_config,
    ):
        self.queue = queue
        self.db_session_maker = db_session_maker
        self.ensemble = ensemble_model
        self.autoencoder = autoencoder_model
        self.rule_config = rule_config

        # History in-memory store: icao24 -> list of previous states (newest first)
        self.history: Dict[str, List[Dict[str, Any]]] = {}
        self._geofence_cache: list[Geofence] = []
        self._geofence_cache_at = 0.0

    async def start_detection_loop(self) -> None:
        """Continuously pulls normalized flight vectors from the ingestion queue."""
        logger.info("Starting detection service processing loop...")
        while True:
            record = await self.queue.get()
            try:
                await self.process_record(record)
            except Exception as e:
                logger.error(
                    f"Error encountered in detection processing loop: {e}",
                    exc_info=True,
                )
            finally:
                self.queue.task_done()

    async def process_record(self, record: Dict[str, Any]) -> None:
        """Evaluates physical rules and ML models on a single flight state update."""
        started_at = time.perf_counter()
        self.rule_config = active_rule_config
        icao24 = record["icao24"]
        prev_history = self.history.get(icao24, [])
        prev_record = prev_history[0] if len(prev_history) > 0 else None

        # Suppression logic cross-reference check
        is_suppressed = record.get("metadata", {}).get("is_known_entity", False)
        known_label = record.get("metadata", {}).get("known_entity_label")

        # --- 1. Evaluate Aerodynamic Rules First ---

        # Check Climb Rate
        rule_climb, climb_reason, climb_evidence = check_impossible_climb_rate(
            record["vertical_rate_ms"], self.rule_config
        )

        # Check Altitude-Velocity Mismatch
        rule_alt_vel, alt_vel_reason, alt_vel_evidence = (
            check_altitude_velocity_mismatch(
                record["altitude_m"],
                record["velocity_ms"],
                record["on_ground"],
                self.rule_config,
            )
        )

        # Check Position Jump (Requires previous state)
        rule_jump = False
        jump_reason = None
        jump_evidence: Dict[str, Any] = {}
        if prev_record:
            rule_jump, jump_reason, jump_evidence = check_position_jump(
                current_lat=record["latitude"],
                current_lon=record["longitude"],
                current_time=record["received_at"],
                prev_lat=prev_record["latitude"],
                prev_lon=prev_record["longitude"],
                prev_time=prev_record["received_at"],
                config=self.rule_config,
            )

        # Check Duplicate ICAO (Within 1s, same ICAO, different locations)
        rule_dup = False
        dup_reason = None
        dup_evidence: Dict[str, Any] = {}
        if prev_record:
            rule_dup, dup_reason, dup_evidence = check_duplicate_icao(
                lat_a=record["latitude"],
                lon_a=record["longitude"],
                time_a=record["received_at"],
                lat_b=prev_record["latitude"],
                lon_b=prev_record["longitude"],
                time_b=prev_record["received_at"],
                config=self.rule_config,
            )

        rule_flags = [rule_jump, rule_dup, rule_climb, rule_alt_vel]
        reasons = [
            r
            for r in [jump_reason, dup_reason, climb_reason, alt_vel_reason]
            if r is not None
        ]

        squawk = str(record.get("squawk") or "").strip()
        emergency_code = squawk if squawk in EMERGENCY_SQUAWKS else None
        rapid_descent = bool(
            not record.get("on_ground")
            and record.get("vertical_rate_ms") is not None
            and record["vertical_rate_ms"] <= -12.7
        )
        if prev_record and not record.get("on_ground"):
            elapsed = (
                record["received_at"] - prev_record["received_at"]
            ).total_seconds()
            if elapsed > 0:
                rapid_descent = (
                    rapid_descent
                    or (record["altitude_m"] - prev_record["altitude_m"]) / elapsed
                    <= -12.7
                )
        if emergency_code:
            reasons.append(EMERGENCY_SQUAWKS[emergency_code])
        if rapid_descent:
            reasons.append(
                "Rapid descent detected (at least 2,500 ft/min from live telemetry)"
            )

        planned_route = record.get("planned_route")
        route_deviation_nm = (
            _distance_to_route_nm(
                record["latitude"], record["longitude"], planned_route
            )
            if planned_route
            else None
        )
        route_deviation = route_deviation_nm is not None and route_deviation_nm > float(
            record.get("route_tolerance_nm", 10.0)
        )
        if route_deviation:
            reasons.append(
                f"Route deviation: {route_deviation_nm:.1f} NM from supplied filed route"
            )

        # Use operator-configured zones from the database; an initial position inside
        # a zone is not treated as a crossing until a prior outside observation exists.
        geofence_entries: list[str] = []
        geofences_inside: list[int] = []
        if time.monotonic() - self._geofence_cache_at > 10:
            async with self.db_session_maker() as session:
                result = await session.execute(
                    select(Geofence).where(Geofence.enabled.is_(True))
                )
                refreshed_geofences = list(result.scalars().all())
                # Re-evaluate each aircraft's last known point against a changed
                # zone set, so adding a zone does not falsely alert on aircraft
                # already inside it before the zone was configured.
                for history in self.history.values():
                    if not history:
                        continue
                    previous = history[0]
                    previously_inside: list[int] = []
                    for zone in refreshed_geofences:
                        lat_delta = radians(previous["latitude"] - zone.latitude)
                        lon_delta = radians(previous["longitude"] - zone.longitude)
                        lat_a, lat_b = radians(zone.latitude), radians(
                            previous["latitude"]
                        )
                        haversine = (
                            sin(lat_delta / 2) ** 2
                            + cos(lat_a) * cos(lat_b) * sin(lon_delta / 2) ** 2
                        )
                        distance_km = (
                            6371.0088
                            * 2
                            * atan2(sqrt(haversine), sqrt(max(0.0, 1.0 - haversine)))
                        )
                        if distance_km <= zone.radius_km:
                            previously_inside.append(zone.id)
                    previous["geofences_inside"] = previously_inside
                self._geofence_cache = refreshed_geofences
                self._geofence_cache_at = time.monotonic()
        active_geofences = self._geofence_cache
        previous_inside = set((prev_record or {}).get("geofences_inside", []))
        for zone in active_geofences:
            lat_delta = radians(record["latitude"] - zone.latitude)
            lon_delta = radians(record["longitude"] - zone.longitude)
            lat_a, lat_b = radians(zone.latitude), radians(record["latitude"])
            haversine = (
                sin(lat_delta / 2) ** 2
                + cos(lat_a) * cos(lat_b) * sin(lon_delta / 2) ** 2
            )
            distance_km = (
                6371.0088 * 2 * atan2(sqrt(haversine), sqrt(max(0.0, 1.0 - haversine)))
            )
            if distance_km <= zone.radius_km:
                geofences_inside.append(zone.id)
                if prev_record is not None and zone.id not in previous_inside:
                    geofence_entries.append(zone.name)
        if geofence_entries:
            reasons.extend(
                f"Entered configured geofence: {name}" for name in geofence_entries
            )
        record["geofences_inside"] = geofences_inside

        # --- 2. Calculate Rolling Window Features ---
        states = [record] + prev_history[:4]
        speeds = [s["velocity_ms"] for s in states]
        headings = [s["heading_deg"] for s in states]
        vert_rates = [s["vertical_rate_ms"] for s in states]

        speed_var = float(np.var(speeds)) if len(speeds) > 1 else 0.0
        heading_var = float(np.var(headings)) if len(headings) > 1 else 0.0
        alt_rate_var = float(np.var(vert_rates)) if len(vert_rates) > 1 else 0.0

        time_diff = 0.0
        if prev_record:
            time_diff = (
                record["received_at"] - prev_record["received_at"]
            ).total_seconds()

        feature_vector = np.array(
            [
                speed_var,
                heading_var,
                alt_rate_var,
                time_diff,
                float(rule_jump),
                float(rule_dup),
                float(rule_climb),
                float(rule_alt_vel),
            ]
        )

        # --- 3. Evaluate Machine Learning Models ---

        # Soft-Voting Ensemble (supervised anomaly prediction)
        ensemble_score, shap_explanation = self.ensemble.predict_anomaly(feature_vector)

        # PyTorch Autoencoder (unsupervised reconstruction error)
        ae_features = np.array([speed_var, heading_var, alt_rate_var, time_diff])
        ae_score = self.autoencoder.compute_anomaly_score(ae_features)

        # Trilateration check
        sensors = record.get("sensors") or []
        trilateration_score, tri_reason, tri_evidence = (
            check_trilateration_plausibility(
                record["latitude"], record["longitude"], sensors
            )
        )
        if tri_reason != "consistent" and tri_reason != "inconclusive":
            reasons.append(tri_reason)

        # --- 4. Combine Scoring Signals ---
        combined_risk_score, is_alert_triggered = combine_scores(
            rule_flags=rule_flags,
            ensemble_score=ensemble_score,
            autoencoder_score=ae_score,
            trilateration_consistency=trilateration_score,
            threshold=0.7,
        )
        if emergency_code:
            combined_risk_score = max(
                combined_risk_score, 0.98 if emergency_code == "7500" else 0.9
            )
        if rapid_descent or route_deviation:
            combined_risk_score = max(
                combined_risk_score, 0.82 if rapid_descent and route_deviation else 0.72
            )
        if geofence_entries:
            combined_risk_score = max(combined_risk_score, 0.8)
        is_alert_triggered = bool(
            is_alert_triggered
            or emergency_code
            or rapid_descent
            or route_deviation
            or geofence_entries
        )
        operational_flags = [
            FEATURE_NAMES[i] for i, flag in enumerate(rule_flags) if flag
        ]
        if emergency_code:
            operational_flags.append(f"emergency_squawk_{emergency_code}")
        if rapid_descent:
            operational_flags.append("rapid_descent")
        if route_deviation:
            operational_flags.append("filed_route_deviation")
        if geofence_entries:
            operational_flags.extend(
                f"geofence_entry:{name}" for name in geofence_entries
            )

        detection_signature = "|".join(sorted(operational_flags))
        new_detection = bool(
            is_alert_triggered
            and (
                not prev_record
                or prev_record.get("detection_signature") != detection_signature
            )
        )
        record["detection_signature"] = detection_signature
        record["emergency_code"] = emergency_code
        record["rapid_descent"] = rapid_descent
        record["route_deviation_nm"] = (
            round(route_deviation_nm, 2) if route_deviation_nm is not None else None
        )
        live_state = await live_registry.upsert(
            record,
            risk_score=combined_risk_score,
            rule_flags=operational_flags,
        )
        runtime_metrics.packets_processed += 1
        runtime_metrics.last_event_at = datetime.now(timezone.utc)
        runtime_metrics.last_processing_ms = round(
            (time.perf_counter() - started_at) * 1000, 2
        )
        try:
            from app.api.v1.endpoints import manager as ws_manager

            await ws_manager.broadcast(
                {
                    "event": "AIRCRAFT_UPDATE",
                    "payload": live_state,
                }
            )
        except Exception:
            logger.debug("Unable to broadcast live aircraft update", exc_info=True)

        # --- 5. Generate Audit Trail Logs ---
        audit_payload = {
            "icao24": icao24,
            "callsign": record["callsign"],
            "combined_risk_score": combined_risk_score,
            "rules_triggered": operational_flags,
            "ensemble_score": ensemble_score,
            "autoencoder_score": ae_score,
            "trilateration_consistency": trilateration_score,
            "is_known_entity": is_suppressed,
            "known_entity_label": known_label,
            "alert_triggered": bool(
                new_detection and (not is_suppressed or emergency_code)
            ),
        }

        if is_suppressed and not emergency_code:
            logger.info(
                json.dumps(
                    {
                        "event": "AUDIT_DECISION_SUPPRESSED",
                        "payload": audit_payload,
                        "message": f"[AUDIT] Decision: SUPPRESSED for known entity {icao24} ({known_label}). Calculated risk: {combined_risk_score:.2f}.",
                    }
                )
            )
        else:
            decision = "ALERT" if new_detection else "PASS"
            logger.info(
                json.dumps(
                    {
                        "event": f"AUDIT_DECISION_{decision}",
                        "payload": audit_payload,
                        "message": f"[AUDIT] Decision: {decision} for aircraft {icao24}. Risk: {combined_risk_score:.2f}.",
                    }
                )
            )

        # Update local rolling state history
        self.history[icao24] = [record] + prev_history[:9]  # Keep last 10 states

        # --- 6. Write to Database ---
        state_id = None
        try:
            async with self.db_session_maker() as session:
                db_state = AircraftState(
                    icao24=record["icao24"],
                    callsign=record["callsign"],
                    latitude=record["latitude"],
                    longitude=record["longitude"],
                    altitude_m=record["altitude_m"],
                    velocity_ms=record["velocity_ms"],
                    heading_deg=record["heading_deg"],
                    vertical_rate_ms=record["vertical_rate_ms"],
                    on_ground=record["on_ground"],
                    squawk=record.get("squawk"),
                    received_at=record["received_at"],
                    source=record["source"],
                )
                session.add(db_state)
                await session.commit()
                await session.refresh(db_state)
                state_id = db_state.id
        except Exception as e:
            logger.error(f"Failed to record AircraftState to database: {e}")

        # Save Alert if triggered and NOT suppressed
        if (
            new_detection
            and (not is_suppressed or emergency_code)
            and state_id is not None
        ):
            try:
                reason_text = (
                    "; ".join(reasons)
                    if len(reasons) > 0
                    else "Anomaly detected by combined risk score."
                )
                evidence_data = {
                    "rule_flags": {
                        "position_jump": jump_evidence,
                        "duplicate_icao": dup_evidence,
                        "climb_rate": climb_evidence,
                        "alt_vel_mismatch": alt_vel_evidence,
                    },
                    "trilateration": tri_evidence,
                    "model_scores": {
                        "ensemble_score": ensemble_score,
                        "autoencoder_score": ae_score,
                    },
                }

                async with self.db_session_maker() as session:
                    db_alert = Alert(
                        icao24=record["icao24"],
                        aircraft_state_id=state_id,
                        rule_flags=operational_flags,
                        ensemble_score=ensemble_score,
                        autoencoder_score=ae_score,
                        combined_risk_score=combined_risk_score,
                        reason_text=reason_text,
                        shap_explanation={
                            "shap": shap_explanation,
                            "evidence": evidence_data,
                        },
                        detected_at=datetime.now(timezone.utc),
                        is_synthetic=False,
                        acknowledged=False,
                    )
                    session.add(db_alert)
                    await session.commit()
                    await session.refresh(db_alert)
                    alert_id = db_alert.id
                logger.info(
                    f"Successfully recorded Alert for aircraft {icao24} in database."
                )
                runtime_metrics.alerts_triggered += 1

                # Broadcast alert in real-time via WebSocket
                try:
                    from app.api.v1.endpoints import manager as ws_manager
                    from app.core.realtime import incidents

                    event_time = datetime.now(timezone.utc).isoformat()
                    timeline_event = {
                        "at": event_time,
                        "type": "telemetry_detection",
                        "signals": operational_flags,
                        "squawk": squawk or None,
                        "altitude_m": record["altitude_m"],
                        "vertical_rate_ms": record["vertical_rate_ms"],
                        "route_deviation_nm": (
                            round(route_deviation_nm, 2)
                            if route_deviation_nm is not None
                            else None
                        ),
                        "risk_score": round(combined_risk_score, 4),
                        "source": record.get("source", "unknown"),
                    }
                    current_incident = next(
                        (
                            incident
                            for incident in incidents.values()
                            if incident.get("icao24") == record["icao24"]
                            and incident.get("status") not in {"resolved", "archived"}
                        ),
                        None,
                    )
                    incident_id = (
                        current_incident["id"]
                        if current_incident
                        else f"{record['icao24']}-{int(datetime.now(timezone.utc).timestamp() * 1000)}"
                    )
                    if current_incident:
                        current_incident["timeline"].append(timeline_event)
                        current_incident["updated_at"] = event_time
                        current_incident["risk_score"] = max(
                            current_incident["risk_score"],
                            round(combined_risk_score, 4),
                        )
                        current_incident["rule_flags"] = sorted(
                            set(current_incident["rule_flags"] + operational_flags)
                        )
                        incident = current_incident
                    else:
                        incident = {
                            "id": incident_id,
                            "icao24": record["icao24"],
                            "status": "new",
                            "created_at": event_time,
                            "risk_score": round(combined_risk_score, 4),
                            "reason": reason_text,
                            "rule_flags": operational_flags,
                            "comments": [],
                            "timeline": [timeline_event],
                            "source": record.get("source", "unknown"),
                        }
                        incidents[incident_id] = incident
                    async with self.db_session_maker() as session:
                        persisted_result = await session.execute(
                            select(Incident)
                            .where(
                                Incident.icao24 == record["icao24"],
                                Incident.status.notin_(["resolved", "archived"]),
                            )
                            .order_by(Incident.created_at.desc())
                            .limit(1)
                        )
                        persisted_incident = persisted_result.scalar_one_or_none()
                        if persisted_incident:
                            persisted_incident.timeline = list(
                                persisted_incident.timeline or []
                            ) + [timeline_event]
                            persisted_incident.rule_flags = sorted(
                                set(
                                    (persisted_incident.rule_flags or [])
                                    + operational_flags
                                )
                            )
                            persisted_incident.risk_score = max(
                                persisted_incident.risk_score,
                                round(combined_risk_score, 4),
                            )
                            persisted_incident.updated_at = datetime.now(timezone.utc)
                            incident_id = persisted_incident.id
                        else:
                            persisted_incident = Incident(
                                id=incident_id,
                                icao24=record["icao24"],
                                status="new",
                                created_at=datetime.fromisoformat(event_time),
                                updated_at=datetime.fromisoformat(event_time),
                                risk_score=round(combined_risk_score, 4),
                                reason=reason_text,
                                rule_flags=operational_flags,
                                comments=[],
                                timeline=[timeline_event],
                                source=record.get("source", "unknown"),
                            )
                            session.add(persisted_incident)
                        await session.commit()
                        incident["id"] = incident_id
                    await live_registry.append_event(record["icao24"], timeline_event)
                    await ws_manager.broadcast(
                        {
                            "event": "ALERT_TRIGGERED",
                            "incident_id": incident_id,
                            "icao24": record["icao24"],
                            "combined_risk_score": combined_risk_score,
                            "reason_text": reason_text,
                            "rule_flags": operational_flags,
                            "trust_score": round(
                                max(0.0, 1.0 - combined_risk_score) * 100, 1
                            ),
                            "detected_at": event_time,
                            "timeline": incident["timeline"],
                            "alert_id": alert_id,
                            "model_scores": evidence_data["model_scores"],
                            "evidence": evidence_data,
                        }
                    )
                except Exception:
                    pass
            except Exception as e:
                logger.error(f"Failed to record Alert to database: {e}")
