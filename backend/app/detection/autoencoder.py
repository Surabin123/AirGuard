import math
import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple

# --- Ground Receivers Reference Database ---
# Mapped coordinates for ground receiver station nodes
MOCK_RECEIVERS = {
    "1": (37.7749, -122.4194),  # San Francisco
    "2": (37.8044, -122.2712),  # Oakland
    "3": (37.3382, -121.8863),  # San Jose
    "4": (38.5816, -121.4944),  # Sacramento
    "5": (36.7783, -119.4179),  # Fresno
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "autoencoder.pth")

# --- Autoencoder Network Architecture ---


class AutoencoderModel(nn.Module):
    def __init__(self, input_dim: int = 4, latent_dim: int = 3):
        super(AutoencoderModel, self).__init__()
        # Compression layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8), nn.ReLU(), nn.Linear(8, latent_dim)
        )
        # Reconstruction layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8), nn.ReLU(), nn.Linear(8, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class UnsupervisedAutoencoder:
    def __init__(self, input_dim: int = 4, latent_dim: int = 3):
        self.model = AutoencoderModel(input_dim, latent_dim)
        self.model.eval()
        self.load_weights()

    def load_weights(self) -> None:
        """Loads weights from disk if trained file exists."""
        if os.path.exists(MODEL_PATH):
            try:
                self.model.load_state_dict(torch.load(MODEL_PATH))
                self.model.eval()
            except Exception:
                pass

    def compute_anomaly_score(self, features: np.ndarray) -> float:
        """Calculate reconstruction MSE and scale to 0-1 probability score.

        Sigmoid-like scaling maps the range [0, inf) to [0, 1].
        """
        tensor_features = torch.FloatTensor(features.reshape(1, -1))
        with torch.no_grad():
            reconstructed = self.model(tensor_features)
            mse = torch.mean((tensor_features - reconstructed) ** 2).item()

        # Scale MSE to [0, 1] range using an exponential decay curve.
        # MSE of 0.0 -> score 0.0. High MSE -> asymptotes to 1.0.
        anomaly_score = 1.0 - math.exp(-mse / 5.0)
        return anomaly_score


# --- Trilateration Plausibility Check ---


def check_trilateration_plausibility(
    aircraft_lat: float, aircraft_lon: float, sensors: List[str]
) -> Tuple[float, str, Dict[str, Any]]:
    """Verify that the reporting sensors are within physically plausible range.

    If sensor locations are known, checks the distance from the target's reported position
    to each receiver. A spoofed transmitter on the ground broadcasting false aircraft
    positions cannot satisfy the geometric ranges of multiple receivers.
    """
    # Exclude invalid or empty sensor values
    valid_sensors = [s for s in sensors if s in MOCK_RECEIVERS]

    # Degrade gracefully if data is sparse
    if len(valid_sensors) < 2:
        return (
            1.0,
            "inconclusive",
            {"reason": "insufficient sensor data", "sensors_evaluated": valid_sensors},
        )

    # Haversine distance calculator
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        p1 = math.radians(lat1)
        p2 = math.radians(lat2)
        dp = math.radians(lat2 - lat1)
        dl = math.radians(lon2 - lon1)
        a = (math.sin(dp / 2.0) ** 2) + math.cos(p1) * math.cos(p2) * (
            math.sin(dl / 2.0) ** 2
        )
        return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    distances = []
    for s_id in valid_sensors:
        rx_lat, rx_lon = MOCK_RECEIVERS[s_id]
        dist = haversine(rx_lat, rx_lon, aircraft_lat, aircraft_lon)
        distances.append(dist)

    max_dist = max(distances)
    avg_dist = sum(distances) / len(distances)

    evidence = {
        "sensors_evaluated": valid_sensors,
        "distances_km": distances,
        "max_distance_km": max_dist,
        "avg_distance_km": avg_dist,
    }

    # Physical Boundary: Line-of-sight limit for typical ADS-B ground stations is ~350 km.
    # If any reporting sensor is > 350 km away from the reported coordinates, the signal geometry is implausible.
    if max_dist > 350.0:
        # Consistency drops off linearly beyond the 350 km threshold
        consistency = max(0.0, 1.0 - (max_dist - 350.0) / 150.0)
        reason = f"physically inconsistent geometry: max receiver distance of {max_dist:.1f} km exceeds line-of-sight limits."
        return consistency, reason, evidence

    return 1.0, "consistent", evidence


# --- Combined Scoring Logic ---


def combine_scores(
    rule_flags: List[bool],
    ensemble_score: float,
    autoencoder_score: float,
    trilateration_consistency: float,
    threshold: float = 0.7,
) -> Tuple[float, bool]:
    """Combines rule, ensemble, autoencoder, and trilateration signals into a risk score.

    Weighted Scoring Formula:
    - Rule Risk (W_rules = 0.40): Evaluates if any explicit physical boundaries (Prompt 5) were violated.
    - Ensemble Classifier (W_ensemble = 0.30): Supervised machine learning prediction.
    - Autoencoder Reconstruction (W_autoencoder = 0.20): Unsupervised deep learning anomaly signal.
    - Trilateration Inconsistency (W_trilateration = 0.10): Geometric consistency (1.0 - consistency).

    Score Summation:
        combined_risk = 0.40 * (any(rule_flags))
                        + 0.30 * ensemble_score
                        + 0.20 * autoencoder_score
                        + 0.10 * (1.0 - trilateration_consistency)

    Returns:
        combined_risk_score (0.0 to 1.0)
        is_alert_triggered (True if combined_risk >= threshold)
    """
    rule_risk = 1.0 if any(rule_flags) else 0.0
    trilateration_inconsistency = 1.0 - trilateration_consistency

    # Define weights
    w_rules = 0.40
    w_ensemble = 0.30
    w_autoencoder = 0.20
    w_trilateration = 0.10

    # Calculate weighted combined risk score
    combined_risk = (
        (w_rules * rule_risk)
        + (w_ensemble * ensemble_score)
        + (w_autoencoder * autoencoder_score)
        + (w_trilateration * trilateration_inconsistency)
    )

    # Ensure score is strictly bounded to [0.0, 1.0]
    combined_risk = min(1.0, max(0.0, combined_risk))

    is_triggered = combined_risk >= threshold
    return combined_risk, is_triggered
