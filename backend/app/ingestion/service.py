import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import httpx
from sqlalchemy import select

from app.models import KnownEntity
from app.core.realtime import runtime_metrics

logger = logging.getLogger("airguard.ingestion")


class CircuitBreakerOpenException(Exception):
    pass


class OpenSkyIngestionService:
    def __init__(
        self,
        queue: asyncio.Queue,
        db_session_maker: Any,
        poll_interval_seconds: float = 8.0,
        opensky_url: str = "https://opensky-network.org/api/states/all",
        max_retries: int = 5,
        cooldown_seconds: float = 60.0,
    ):
        self.queue = queue
        self.db_session_maker = db_session_maker
        self.poll_interval = poll_interval_seconds
        self.opensky_url = opensky_url

        # Ingestion Client
        self.client = httpx.AsyncClient(timeout=10.0)

        # Known Entities Cache
        self.known_entities: Dict[str, str] = {}  # icao24 -> label

        # Reliability Control State
        self.breaker_state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.consecutive_failures = 0
        self.max_retries = max_retries
        self.cooldown_duration = cooldown_seconds
        self.cooldown_until: float = 0.0
        self.backoff_seconds: float = 0.0
        self.last_successful_poll: Optional[datetime] = None

    async def refresh_known_entities(self) -> None:
        """Fetch known entities from the database and refresh the in-memory cache."""
        try:
            async with self.db_session_maker() as session:
                result = await session.execute(select(KnownEntity))
                entities = result.scalars().all()
                self.known_entities = {e.icao24.lower(): e.label for e in entities}
                logger.info(
                    f"Refreshed known entities cache. Loaded {len(self.known_entities)} records."
                )
        except Exception as e:
            logger.error(f"Failed to refresh known entities from database: {e}")

    def normalize_state(self, vector: List[Any]) -> Optional[Dict[str, Any]]:
        """Normalize OpenSky state vector array into a standardized schema dict.

        OpenSky State Vector Spec:
        - index 0 (icao24): Unique ICAO 24-bit address (str)
        - index 1 (callsign): Callsign of the vehicle (str or null)
        - index 2 (origin_country): Country name of registration (str)
        - index 3 (time_position): Unix epoch of last position update (int or null)
        - index 4 (last_contact): Unix epoch of last transponder signal (int)
        - index 5 (longitude): Geodetic longitude in decimal degrees (float or null)
        - index 6 (latitude): Geodetic latitude in decimal degrees (float or null)
        - index 7 (baro_altitude): Barometric altitude in meters (float or null)
        - index 8 (on_ground): True if taxiing or landed (bool)
        - index 9 (velocity): Ground speed in meters per second (float or null)
        - index 10 (true_track): True track/heading in decimal degrees clockwise from north (float or null)
        - index 11 (vertical_rate): Vertical speed in meters per second (float or null)
        - index 12 (sensors): Sensor IDs (list[int] or null)
        - index 13 (geo_altitude): Geometric altitude in meters (float or null)
        - index 14 (squawk): Transponder squawk code (str or null)
        - index 15 (spi): Special purpose indicator (bool)
        - index 16 (position_source): Position source identifier (int)
        """
        # Critical Filter: We require latitude and longitude to track/score aircraft
        if len(vector) < 7 or vector[5] is None or vector[6] is None:
            return None

        icao24 = str(vector[0]).strip().lower()
        callsign = str(vector[1]).strip() if vector[1] is not None else None

        # Unit Conversions & Null Handling:
        # - Latitude/Longitude: decimal degrees (No conversion, stored directly)
        lat = float(vector[6])
        lng = float(vector[5])

        # - Altitude: OpenSky delivers altitude in meters.
        #   We use baro_altitude (index 7) as primary, falling back to geo_altitude (index 13).
        #   If both are null, we set to 0.0 and document default.
        altitude = 0.0
        if vector[7] is not None:
            altitude = float(vector[7])
        elif vector[13] is not None:
            altitude = float(vector[13])

        # - Velocity: OpenSky ground speed is delivered in m/s. Stored directly. Default to 0.0.
        velocity = float(vector[9]) if vector[9] is not None else 0.0

        # - Heading: OpenSky track angle is in decimal degrees from North. Default to 0.0.
        heading = float(vector[10]) if vector[10] is not None else 0.0

        # - Vertical Rate: OpenSky vertical speed is in m/s. Stored directly. Default to 0.0.
        vertical_rate = float(vector[11]) if vector[11] is not None else 0.0

        on_ground = bool(vector[8])

        # - Received timestamp: Convert unix epoch to timezone-aware datetime.
        #   Fallback sequence: time_position (index 3) -> last_contact (index 4) -> current system time.
        timestamp_epoch = vector[3] if vector[3] is not None else vector[4]
        if timestamp_epoch is not None:
            received_at = datetime.fromtimestamp(timestamp_epoch, tz=timezone.utc)
        else:
            received_at = datetime.now(timezone.utc)

        # Cross-reference with Known Entities cache
        known_label = self.known_entities.get(icao24)
        is_known = known_label is not None

        if is_known:
            # Explicitly log suppression tag
            logger.info(
                json.dumps(
                    {
                        "event": "SUPPRESSION",
                        "icao24": icao24,
                        "label": known_label,
                        "message": f"[SUPPRESSION] Target flagged as known entity: {known_label}. Downstream checks bypassed.",
                    }
                )
            )

        return {
            "icao24": icao24,
            "callsign": callsign,
            "latitude": lat,
            "longitude": lng,
            "altitude_m": altitude,
            "velocity_ms": velocity,
            "heading_deg": heading,
            "vertical_rate_ms": vertical_rate,
            "on_ground": on_ground,
            "squawk": (
                str(vector[14]).strip()
                if len(vector) > 14 and vector[14] is not None
                else None
            ),
            "spi": (
                bool(vector[15])
                if len(vector) > 15 and vector[15] is not None
                else False
            ),
            "received_at": received_at,
            "source": "opensky",
            "metadata": {
                "is_known_entity": is_known,
                "known_entity_label": known_label,
            },
        }

    def update_circuit_state_on_failure(self) -> None:
        """Increment failure state, calculate backoff, and trip breaker if threshold hit."""
        self.consecutive_failures += 1

        if self.consecutive_failures >= self.max_retries:
            self.breaker_state = "OPEN"
            self.cooldown_until = time.time() + self.cooldown_duration
            self.backoff_seconds = 0.0
            logger.warning(
                json.dumps(
                    {
                        "event": "CIRCUIT_TRIPPED",
                        "breaker_state": self.breaker_state,
                        "cooldown_until": self.cooldown_until,
                        "consecutive_failures": self.consecutive_failures,
                        "message": f"Circuit breaker tripped to OPEN. Cooldown active for {self.cooldown_duration}s.",
                    }
                )
            )
        else:
            # Exponential backoff base 2s (e.g. 2s, 4s, 8s, 16s)
            self.backoff_seconds = 2.0**self.consecutive_failures
            logger.info(
                json.dumps(
                    {
                        "event": "BACKOFF_ENGAGED",
                        "consecutive_failures": self.consecutive_failures,
                        "backoff_seconds": self.backoff_seconds,
                        "message": f"Polled failed. Engaging exponential backoff for {self.backoff_seconds}s.",
                    }
                )
            )

    def update_circuit_state_on_success(self) -> None:
        """Reset failure counters and restore breaker state to CLOSED."""
        if self.breaker_state != "CLOSED":
            logger.info(
                json.dumps(
                    {
                        "event": "CIRCUIT_CLOSED",
                        "breaker_state": "CLOSED",
                        "message": "Circuit breaker restored to CLOSED state.",
                    }
                )
            )
        self.breaker_state = "CLOSED"
        self.consecutive_failures = 0
        self.backoff_seconds = 0.0

    def evaluate_circuit(self) -> None:
        """Check if circuit is OPEN and test if cooldown expired to transition to HALF_OPEN."""
        now = time.time()

        if self.breaker_state == "OPEN":
            if now >= self.cooldown_until:
                self.breaker_state = "HALF_OPEN"
                logger.info(
                    json.dumps(
                        {
                            "event": "CIRCUIT_HALF_OPEN",
                            "breaker_state": self.breaker_state,
                            "message": "Cooldown expired. Circuit transitioned to HALF_OPEN. Testing next poll.",
                        }
                    )
                )
            else:
                remaining = self.cooldown_until - now
                logger.debug(
                    f"Circuit breaker is OPEN. Cooldown remaining: {remaining:.1f}s"
                )
                raise CircuitBreakerOpenException(
                    f"Circuit breaker is OPEN. Cooldown remaining: {remaining:.1f}s"
                )

    async def poll_api(self) -> List[List[Any]]:
        """Query OpenSky endpoint. Enforces circuit evaluation and metrics tracking."""
        self.evaluate_circuit()

        start_time = time.time()
        try:
            res = await self.client.get(self.opensky_url)
            latency = (time.time() - start_time) * 1000.0  # ms
            runtime_metrics.last_poll_latency_ms = round(latency, 2)

            if res.status_code != 200:
                raise httpx.HTTPStatusError(
                    f"Non-200 response: {res.status_code}",
                    request=res.request,
                    response=res,
                )

            data = res.json()
            states = data.get("states") or []

            self.update_circuit_state_on_success()
            self.last_successful_poll = datetime.now(timezone.utc)

            # Log structured poll metric
            logger.info(
                json.dumps(
                    {
                        "event": "POLL_METRICS",
                        "success": True,
                        "record_count": len(states),
                        "latency_ms": latency,
                        "breaker_state": self.breaker_state,
                        "backoff_seconds": self.backoff_seconds,
                    }
                )
            )
            return states

        except Exception as e:
            latency = (time.time() - start_time) * 1000.0  # ms
            runtime_metrics.last_poll_latency_ms = round(latency, 2)
            logger.error(
                json.dumps(
                    {
                        "event": "POLL_METRICS",
                        "success": False,
                        "error": str(e),
                        "latency_ms": latency,
                        "breaker_state": self.breaker_state,
                        "backoff_seconds": self.backoff_seconds,
                    }
                )
            )
            self.update_circuit_state_on_failure()
            raise

    async def run_single_poll_cycle(self) -> int:
        """Executes a single fetch, normalizes vectors, and pushes matching records to Queue."""
        # Wait backoff if engaged
        if self.backoff_seconds > 0:
            await asyncio.sleep(self.backoff_seconds)

        try:
            raw_states = await self.poll_api()
            runtime_metrics.packets_received += len(raw_states)
            normalized_count = 0

            for vector in raw_states:
                normalized = self.normalize_state(vector)
                if normalized is not None:
                    await self.queue.put(normalized)
                    normalized_count += 1
            runtime_metrics.packets_dropped += max(
                0, len(raw_states) - normalized_count
            )
            runtime_metrics.last_event_at = datetime.now(timezone.utc)
            runtime_metrics.source_status["opensky"] = "healthy"

            return normalized_count
        except CircuitBreakerOpenException:
            # Let the loop wait and retry later
            return 0
        except Exception:
            # Already logged in poll_api
            return 0

    async def start_polling_loop(self) -> None:
        """Spins up the continuous async polling run loop."""
        logger.info("Starting OpenSky Ingestion polling loop...")
        await self.refresh_known_entities()

        while True:
            await self.run_single_poll_cycle()
            await asyncio.sleep(self.poll_interval)
