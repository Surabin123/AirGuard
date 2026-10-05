import pytest
from datetime import datetime, timezone
from sqlalchemy import text
from app.core.database import Base, engine, async_session_maker
from app.models import AircraftState, Alert, ModelRun, KnownEntity

# --- Static Model Tests (Always Run, No DB required) ---


def test_static_aircraft_state_schema():
    """Verify AircraftState schema columns statically via SQLAlchemy metadata."""
    table = AircraftState.__table__

    assert table.c.id.type.__class__.__name__ == "BigInteger"
    assert table.c.icao24.type.length == 6
    assert table.c.callsign.type.length == 10
    assert table.c.latitude.type.__class__.__name__ == "Float"
    assert table.c.longitude.type.__class__.__name__ == "Float"
    assert table.c.altitude_m.type.__class__.__name__ == "Float"
    assert table.c.velocity_ms.type.__class__.__name__ == "Float"
    assert table.c.heading_deg.type.__class__.__name__ == "Float"
    assert table.c.vertical_rate_ms.type.__class__.__name__ == "Float"
    assert table.c.on_ground.type.__class__.__name__ == "Boolean"
    assert table.c.received_at.type.timezone is True
    assert table.c.source.server_default.arg == "opensky"

    # Check composite index
    composite_index = next(
        (idx for idx in table.indexes if idx.name == "idx_states_icao_received_desc"),
        None,
    )
    assert (
        composite_index is not None
    ), "Composite index idx_states_icao_received_desc not defined"

    # Verify index columns
    col_names = [c.name for c in composite_index.columns]
    assert "icao24" in col_names
    assert "received_at" in col_names


def test_static_alert_schema():
    """Verify Alert schema columns statically via SQLAlchemy metadata."""
    table = Alert.__table__

    assert table.c.id.type.__class__.__name__ == "BigInteger"
    assert table.c.icao24.type.length == 6
    assert table.c.aircraft_state_id.type.__class__.__name__ == "BigInteger"
    assert table.c.rule_flags.type.__class__.__name__ == "ARRAY"
    assert table.c.rule_flags.type.item_type.__class__.__name__ == "Text"
    assert table.c.ensemble_score.type.__class__.__name__ == "Float"
    assert table.c.autoencoder_score.type.__class__.__name__ == "Float"
    assert table.c.combined_risk_score.type.__class__.__name__ == "Float"
    assert table.c.reason_text.type.__class__.__name__ == "Text"
    assert table.c.shap_explanation.type.__class__.__name__ == "JSONB"
    assert table.c.detected_at.type.timezone is True
    assert table.c.is_synthetic.server_default.arg == "false"
    assert table.c.acknowledged.server_default.arg == "false"

    # Verify foreign key constraint linking to aircraft_states
    fks = list(table.c.aircraft_state_id.foreign_keys)
    assert len(fks) == 1
    assert fks[0].column.table.name == "aircraft_states"
    assert fks[0].column.name == "id"


def test_static_model_run_schema():
    """Verify ModelRun schema columns statically."""
    table = ModelRun.__table__

    assert table.c.id.type.__class__.__name__ == "Integer"
    assert table.c.run_at.type.timezone is True
    assert table.c.model_version.type.length == 20
    assert table.c.true_positives.type.__class__.__name__ == "Integer"
    assert table.c.false_positives.type.__class__.__name__ == "Integer"
    assert table.c.true_negatives.type.__class__.__name__ == "Integer"
    assert table.c.false_negatives.type.__class__.__name__ == "Integer"
    assert table.c.precision.type.__class__.__name__ == "Float"
    assert table.c.recall.type.__class__.__name__ == "Float"
    assert table.c.f1.type.__class__.__name__ == "Float"
    assert table.c.notes.type.__class__.__name__ == "Text"


def test_static_known_entity_schema():
    """Verify KnownEntity schema columns statically."""
    table = KnownEntity.__table__

    assert table.c.id.type.__class__.__name__ == "Integer"
    assert table.c.icao24.type.length == 6
    assert table.c.icao24.unique is True
    assert table.c.label.type.length == 50
    assert table.c.source.type.length == 50
    assert table.c.added_at.type.timezone is True


# --- Live DB Integration Tests (Skipped if DB connection fails) ---


async def check_db_connection() -> bool:
    """Helper to check if the database engine is reachable."""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@pytest.mark.asyncio
async def test_live_db_crud_operations():
    """Run live database migrations, inserts, queries and deletes if connection is active."""
    if not await check_db_connection():
        pytest.skip(
            "PostgreSQL database is offline. Skipping live DB integration test."
        )

    # Recreate tables dynamically in tests
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    async with async_session_maker() as session:
        # 1. Insert AircraftState
        state = AircraftState(
            icao24="A1B2C3",
            callsign="TEST101",
            latitude=37.7749,
            longitude=-122.4194,
            altitude_m=10000.0,
            velocity_ms=250.0,
            heading_deg=180.0,
            vertical_rate_ms=0.0,
            on_ground=False,
            received_at=datetime.now(timezone.utc),
        )
        session.add(state)
        await session.commit()
        await session.refresh(state)

        assert state.id is not None

        # 2. Insert Alert associated with the state
        alert = Alert(
            icao24="A1B2C3",
            aircraft_state_id=state.id,
            rule_flags=["GPS_SPOOF", "SPEED_LIMIT"],
            ensemble_score=0.85,
            autoencoder_score=0.92,
            combined_risk_score=0.89,
            reason_text="Anomalous flight velocity detected",
            shap_explanation={"velocity_ms": 3.4, "altitude_m": -1.2},
            detected_at=datetime.now(timezone.utc),
        )
        session.add(alert)
        await session.commit()
        await session.refresh(alert)

        assert alert.id is not None
        assert alert.aircraft_state.callsign == "TEST101"

        # 3. Clean up
        await session.delete(alert)
        await session.delete(state)
        await session.commit()
