from datetime import datetime, timedelta, timezone

from app.detection.rules import (
    RuleConfig,
    check_position_jump,
    check_duplicate_icao,
    check_impossible_climb_rate,
    check_altitude_velocity_mismatch,
)

# --- 1. Position Jump Boundary Tests ---


def test_position_jump_boundary():
    config = RuleConfig(max_implied_speed_kmh=1200.0)
    now = datetime.now(timezone.utc)
    one_minute_ago = now - timedelta(minutes=1)

    # Normal speed: San Francisco to Oakland (~15 km in 1 min -> 900 km/h)
    flagged, reason, evidence = check_position_jump(
        current_lat=37.8044,
        current_lon=-122.2712,
        current_time=now,
        prev_lat=37.7749,
        prev_lon=-122.4194,
        prev_time=one_minute_ago,
        config=config,
    )
    assert not flagged
    assert evidence["implied_speed_kmh"] < 1000.0

    # Anomalous speed (teleportation): San Francisco to Los Angeles (~560 km in 1 min -> 33,600 km/h)
    flagged, reason, evidence = check_position_jump(
        current_lat=34.0522,
        current_lon=-118.2437,
        current_time=now,
        prev_lat=37.7749,
        prev_lon=-122.4194,
        prev_time=one_minute_ago,
        config=config,
    )
    assert flagged
    assert "Implied speed" in reason
    assert evidence["implied_speed_kmh"] > 30000.0


# --- 2. Duplicate ICAO Boundary Tests ---


def test_duplicate_icao_boundary():
    config = RuleConfig(duplicate_icao_dist_km=50.0)
    t_base = datetime.now(timezone.utc)

    # Case A: Same second, positions are close (15 km apart) -> No anomaly
    flagged, reason, evidence = check_duplicate_icao(
        lat_a=37.7749,
        lon_a=-122.4194,
        time_a=t_base,
        lat_b=37.8044,
        lon_b=-122.2712,
        time_b=t_base,
        config=config,
    )
    assert not flagged

    # Case B: Same second, positions are far (560 km apart) -> Anomaly!
    flagged, reason, evidence = check_duplicate_icao(
        lat_a=37.7749,
        lon_a=-122.4194,
        time_a=t_base,
        lat_b=34.0522,
        lon_b=-118.2437,
        time_b=t_base,
        config=config,
    )
    assert flagged
    assert "Duplicate ICAO" in reason

    # Case C: Positions are far but reported 5 seconds apart (dt > 1.0s) -> No anomaly
    flagged, reason, evidence = check_duplicate_icao(
        lat_a=37.7749,
        lon_a=-122.4194,
        time_a=t_base,
        lat_b=34.0522,
        lon_b=-118.2437,
        time_b=t_base + timedelta(seconds=5),
        config=config,
    )
    assert not flagged


# --- 3. Climb Rate Boundary Tests ---


def test_climb_rate_boundary():
    config = RuleConfig(max_vertical_rate_ms=50.0)

    # Realistic climb: 25 m/s (~4900 ft/min)
    flagged, reason, evidence = check_impossible_climb_rate(
        vertical_rate_ms=25.0, config=config
    )
    assert not flagged

    # Impossible climb: 55 m/s (~10,800 ft/min)
    flagged, reason, evidence = check_impossible_climb_rate(
        vertical_rate_ms=55.0, config=config
    )
    assert flagged
    assert "Vertical rate" in reason

    # Impossible descent: -60 m/s
    flagged, reason, evidence = check_impossible_climb_rate(
        vertical_rate_ms=-60.0, config=config
    )
    assert flagged


# --- 4. Altitude/Velocity Mismatch Boundary Tests ---


def test_altitude_velocity_mismatch_boundary():
    config = RuleConfig(
        max_ground_altitude_m=100.0, max_ground_speed_ms=77.0, min_flight_speed_ms=20.0
    )

    # Case A: Normal ground taxiing (on_ground=True, alt=10m, speed=15 m/s) -> Normal
    flagged, reason, evidence = check_altitude_velocity_mismatch(
        altitude_m=10.0, velocity_ms=15.0, on_ground=True, config=config
    )
    assert not flagged

    # Case B: On ground but at cruise altitude (on_ground=True, alt=5000m, speed=15 m/s) -> Anomaly!
    flagged, reason, evidence = check_altitude_velocity_mismatch(
        altitude_m=5000.0, velocity_ms=15.0, on_ground=True, config=config
    )
    assert flagged
    assert "reports on_ground=True but has altitude" in reason

    # Case C: On ground but traveling supersonic (on_ground=True, alt=10m, speed=350 m/s) -> Anomaly!
    flagged, reason, evidence = check_altitude_velocity_mismatch(
        altitude_m=10.0, velocity_ms=350.0, on_ground=True, config=config
    )
    assert flagged
    assert "reports on_ground=True but has speed" in reason

    # Case D: Normal flight airborne (on_ground=False, alt=10000m, speed=240 m/s) -> Normal
    flagged, reason, evidence = check_altitude_velocity_mismatch(
        altitude_m=10000.0, velocity_ms=240.0, on_ground=False, config=config
    )
    assert not flagged

    # Case E: Airborne but static on the ground (on_ground=False, alt=0m, speed=5 m/s) -> Anomaly!
    flagged, reason, evidence = check_altitude_velocity_mismatch(
        altitude_m=0.0, velocity_ms=5.0, on_ground=False, config=config
    )
    assert flagged
    assert "reports on_ground=False (airborne) but has zero altitude" in reason


# --- 5. Joint Known Entities Suppression Test ---


def run_mock_detection_pipeline(record, prev_record=None, config=RuleConfig()):
    """Mock handler simulating downstream ingestion logic."""
    # Bypassed if known military/test entity
    if record.get("metadata", {}).get("is_known_entity", False):
        return {
            "flagged": False,
            "suppressed": True,
            "reason": "Known Entity Suppression",
        }

    # Run check impossible climb rate
    flagged, reason, _ = check_impossible_climb_rate(record["vertical_rate_ms"], config)
    if flagged:
        return {"flagged": True, "suppressed": False, "reason": reason}

    return {"flagged": False, "suppressed": False, "reason": None}


def test_joint_known_entities_suppression():
    config = RuleConfig(max_vertical_rate_ms=50.0)

    # Record A: Normal commercial aircraft (is_known_entity=False) exceeding vertical climb
    normal_record = {
        "icao24": "a1b2c3",
        "vertical_rate_ms": 75.0,  # Highly anomalous
        "metadata": {"is_known_entity": False, "known_entity_label": None},
    }
    res = run_mock_detection_pipeline(normal_record, config=config)
    assert res["flagged"]
    assert not res["suppressed"]
    assert "Vertical rate" in res["reason"]

    # Record B: Known Military / Test aircraft (is_known_entity=True) exceeding vertical climb
    military_record = {
        "icao24": "d81234",
        "vertical_rate_ms": 75.0,  # Highly anomalous but suppressed
        "metadata": {"is_known_entity": True, "known_entity_label": "MILITARY_JET_F35"},
    }
    res = run_mock_detection_pipeline(military_record, config=config)
    assert not res["flagged"]
    assert res["suppressed"]
    assert res["reason"] == "Known Entity Suppression"
