# AirGuard Backend

This is the Python backend for AirGuard, a real-time ADS-B trust-scoring ground station.

## Tech Stack
- Python 3.11
- FastAPI
- Poetry
- SQLAlchemy (asyncio) + asyncpg
- slowapi (rate-limiting)
- scikit-learn + PyTorch (anomaly detection models)
# Live emergency monitoring and replay

AirGuard reads live position, altitude, vertical rate, and squawk data from its configured OpenSky feed. Squawk codes 7500, 7600, and 7700 create high-priority detections. Rapid descent is detected from a reported vertical rate or successive altitude observations at or above 2,500 ft/min.

`POST /api/v1/ingest` accepts normalized telemetry from an authorized receiver or flight-data provider. Its optional `planned_route` must be the actual filed route, expressed as an ordered array of `{ "latitude": number, "longitude": number }` waypoints; the optional `route_tolerance_nm` controls the cross-track alert threshold. AirGuard does not infer or invent a filed route. OpenSky's current state-vector feed does not populate this field.

`GET /api/v1/aircraft/{icao24}/replay` returns timestamped observations stored by AirGuard, in event-time order. The dashboard playback uses these records and displays an empty state when no history exists. Incident records and operator timeline updates are stored in PostgreSQL.

Before running with an existing database, apply migrations with `alembic upgrade head`. Diversion-airport recommendations are not enabled: the current deployment has no airport/runway operational-status provider, and nearest-airport distance alone is not enough to recommend a safe diversion.

## Operator geofences and evaluation limits

Circular geofences are operator-defined, persisted horizontal reference zones managed through `/api/v1/geofences` (GET/POST/PATCH/DELETE); they are not official restricted-airspace data and currently do not filter by altitude. The detector refreshes enabled zones every 10 seconds and raises an alert only when a subsequent live aircraft observation crosses into a zone; the first observation after startup or zone creation is treated as a baseline. The dashboard renders these zones on the live radar.

Replay evaluation counts only states with an explicit stored synthetic-anomaly label. It can report recall for those positive examples, but precision, F1, false positives, and true negatives remain unavailable until a trustworthy labeled-normal dataset is added. Unlabeled live aircraft are never assumed to be normal ground truth.
