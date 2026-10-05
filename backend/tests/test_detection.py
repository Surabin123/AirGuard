import json
import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from app.detection.service import DetectionService
from app.detection.rules import RuleConfig


@pytest.mark.asyncio
async def test_detection_service_integration():
    queue = asyncio.Queue()

    # 1. Mock DB Session Maker
    db_session = MagicMock()
    async_db_session = AsyncMock()
    db_session.return_value = async_db_session

    async_db_session.commit = AsyncMock()
    async_db_session.refresh = AsyncMock()

    # Mock refresh to inject database row ID
    def mock_refresh(obj):
        obj.id = 12345
        return None

    async_db_session.refresh.side_effect = mock_refresh

    # 2. Mock Machine Learning Estimators
    ensemble = MagicMock()
    ensemble.predict_anomaly.return_value = (
        0.2,
        {"top_features": [], "base_value": 0.05},
    )

    autoencoder = MagicMock()
    autoencoder.compute_anomaly_score.return_value = 0.1

    # 3. Instantiate service
    service = DetectionService(
        queue=queue,
        db_session_maker=db_session,
        ensemble_model=ensemble,
        autoencoder_model=autoencoder,
        rule_config=RuleConfig(),
    )

    # Mock logger to verify structured log fields
    with patch("app.detection.service.logger") as mock_logger:
        t1 = datetime.now(timezone.utc) - timedelta(minutes=1)
        t2 = datetime.now(timezone.utc)

        # --- Test Sequence 1: Deliberate Position Jump (Trigger alert) ---
        state_1 = {
            "icao24": "a1b2c3",
            "callsign": "UAL824",
            "latitude": 37.7749,
            "longitude": -122.4194,
            "altitude_m": 10000.0,
            "velocity_ms": 250.0,
            "heading_deg": 180.0,
            "vertical_rate_ms": 0.0,
            "on_ground": False,
            "received_at": t1,
            "source": "opensky",
            "metadata": {"is_known_entity": False, "known_entity_label": None},
        }

        # Position jumps 560 km in 1 min (implied speed ~33,600 km/h)
        state_2 = {
            "icao24": "a1b2c3",
            "callsign": "UAL824",
            "latitude": 34.0522,
            "longitude": -118.2437,
            "altitude_m": 10000.0,
            "velocity_ms": 250.0,
            "heading_deg": 180.0,
            "vertical_rate_ms": 0.0,
            "on_ground": False,
            "received_at": t2,
            "source": "opensky",
            "metadata": {"is_known_entity": False, "known_entity_label": None},
        }

        await service.process_record(state_1)
        await service.process_record(state_2)

        # Assert correct Alert created in DB
        added_objects = [args[0] for args, _ in async_db_session.add.call_args_list]
        alerts_added = [
            obj for obj in added_objects if obj.__class__.__name__ == "Alert"
        ]

        assert len(alerts_added) == 1
        alert = alerts_added[0]
        assert alert.icao24 == "a1b2c3"
        assert "rule_position_jump" in alert.rule_flags
        assert "Implied speed" in alert.reason_text

        # Assert correct audit decision logged
        log_payloads = [
            json.loads(args[0]) for args, _ in mock_logger.info.call_args_list if args
        ]
        alert_log = next(
            (p for p in log_payloads if p.get("event") == "AUDIT_DECISION_ALERT"), None
        )

        assert alert_log is not None
        assert alert_log["payload"]["icao24"] == "a1b2c3"
        assert alert_log["payload"]["alert_triggered"] is True

        # Reset mocks for next sequence
        async_db_session.add.reset_mock()
        mock_logger.info.reset_mock()

        # --- Test Sequence 2: Known Entity (Suppressed alert) ---
        state_military = {
            "icao24": "d81234",
            "callsign": "MIL-1",
            "latitude": 37.7749,
            "longitude": -122.4194,
            "altitude_m": 5000.0,
            "velocity_ms": 150.0,
            "heading_deg": 90.0,
            "vertical_rate_ms": 80.0,  # Highly anomalous climb rate (>50 m/s)
            "on_ground": False,
            "received_at": t2,
            "source": "opensky",
            "metadata": {"is_known_entity": True, "known_entity_label": "MILITARY_F35"},
        }

        await service.process_record(state_military)

        # Assert NO Alert added to DB for suppressed target
        added_objects_mil = [args[0] for args, _ in async_db_session.add.call_args_list]
        alerts_added_mil = [
            obj for obj in added_objects_mil if obj.__class__.__name__ == "Alert"
        ]
        assert (
            len(alerts_added_mil) == 0
        ), "Alert was incorrectly written to database for suppressed entity"

        # Assert correct suppression logging
        log_payloads_mil = [
            json.loads(args[0]) for args, _ in mock_logger.info.call_args_list if args
        ]
        suppressed_log = next(
            (
                p
                for p in log_payloads_mil
                if p.get("event") == "AUDIT_DECISION_SUPPRESSED"
            ),
            None,
        )

        assert suppressed_log is not None
        assert suppressed_log["payload"]["icao24"] == "d81234"
        assert suppressed_log["payload"]["is_known_entity"] is True
        assert suppressed_log["payload"]["known_entity_label"] == "MILITARY_F35"
