import os
import json
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
import httpx

from app.ingestion.service import OpenSkyIngestionService, CircuitBreakerOpenException

# Load fixture
FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures/opensky_fixture.json")


def load_fixture():
    with open(FIXTURE_PATH, "r") as f:
        return json.load(f)


@pytest.mark.asyncio
async def test_normalization():
    # Setup test queue
    queue = asyncio.Queue()
    db_session_maker = MagicMock()

    service = OpenSkyIngestionService(queue=queue, db_session_maker=db_session_maker)

    # Pre-populate known entities cache
    service.known_entities = {"d81234": "MILITARY_TARGET_A"}

    fixture_data = load_fixture()
    states = fixture_data["states"]

    # 1. Test UAL824 (normal flight)
    val = service.normalize_state(states[0])
    assert val is not None
    assert val["icao24"] == "a1b2c3"
    assert val["callsign"] == "UAL824"
    assert val["latitude"] == 37.7749
    assert val["longitude"] == -122.4194
    assert val["altitude_m"] == 10000.0
    assert val["velocity_ms"] == 250.0
    assert val["heading_deg"] == 180.0
    assert val["on_ground"] is False
    assert val["metadata"]["is_known_entity"] is False

    # 2. Test DLH452 (null baro altitude, falls back to geo altitude 10050.0)
    val = service.normalize_state(states[1])
    assert val is not None
    assert val["altitude_m"] == 10050.0

    # 3. Test MIL-1 (null velocities, heading -> defaults to 0.0, known entity check)
    val = service.normalize_state(states[2])
    assert val is not None
    assert val["velocity_ms"] == 0.0
    assert val["heading_deg"] == 0.0
    assert val["metadata"]["is_known_entity"] is True
    assert val["metadata"]["known_entity_label"] == "MILITARY_TARGET_A"

    # 4. Test f9232a (missing longitude -> should be skipped)
    val = service.normalize_state(states[3])
    assert val is None

    # 5. Test AAL102 (on ground is true)
    val = service.normalize_state(states[4])
    assert val is not None
    assert val["on_ground"] is True


@pytest.mark.asyncio
async def test_circuit_breaker_and_backoff():
    queue = asyncio.Queue()
    db_session_maker = MagicMock()

    # Initialize with max_retries = 3 and short cooldown = 1s for testing
    service = OpenSkyIngestionService(
        queue=queue,
        db_session_maker=db_session_maker,
        max_retries=3,
        cooldown_seconds=1.0,
    )

    # Mock httpx GET call
    mock_get = AsyncMock()
    service.client.get = mock_get

    # Step 1: Simulate 1st failure (consecutive_failures=1, backoff 2^1 = 2s)
    mock_get.side_effect = httpx.ConnectError("Connection timed out")
    with pytest.raises(httpx.ConnectError):
        await service.poll_api()

    assert service.consecutive_failures == 1
    assert service.breaker_state == "CLOSED"
    assert service.backoff_seconds == 2.0

    # Step 2: Simulate 2nd failure (backoff 2^2 = 4s)
    with pytest.raises(httpx.ConnectError):
        await service.poll_api()

    assert service.consecutive_failures == 2
    assert service.breaker_state == "CLOSED"
    assert service.backoff_seconds == 4.0

    # Step 3: Simulate 3rd failure (consecutive_failures=3 reaches max_retries=3 -> TRIPS BREAKER TO OPEN)
    with pytest.raises(httpx.ConnectError):
        await service.poll_api()

    assert service.consecutive_failures == 3
    assert service.breaker_state == "OPEN"
    assert service.backoff_seconds == 0.0  # Resets backoff during OPEN

    # Step 4: Next poll immediately should raise CircuitBreakerOpenException (cooldown active)
    with pytest.raises(CircuitBreakerOpenException):
        await service.poll_api()

    # Step 5: Wait 1.1s for cooldown to expire
    await asyncio.sleep(1.1)

    # Next poll should change state to HALF_OPEN and run the query.
    # Set the return value to a successful mock response
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {
        "states": [
            [
                "a1b2c3",
                "UAL824",
                "USA",
                1722784490,
                1722784495,
                -122.4194,
                37.7749,
                10000.0,
                False,
                250.0,
                180.0,
                0.0,
                None,
                10100.0,
                "1200",
                False,
                0,
            ]
        ]
    }
    mock_get.side_effect = None
    mock_get.return_value = mock_res

    states = await service.poll_api()
    assert len(states) == 1

    # Breaker should recover to CLOSED and consecutive_failures resets
    assert service.breaker_state == "CLOSED"
    assert service.consecutive_failures == 0
