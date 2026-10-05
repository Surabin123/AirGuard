import numpy as np
import torch

from app.detection.autoencoder import (
    AutoencoderModel,
    UnsupervisedAutoencoder,
    check_trilateration_plausibility,
    combine_scores,
)

# --- 1. PyTorch Autoencoder Structural Tests ---


def test_autoencoder_architecture():
    # Input dim 4, latent dim 3
    model = AutoencoderModel(input_dim=4, latent_dim=3)

    # Assert structural forward pass
    dummy_input = torch.randn(5, 4)  # batch size 5, features 4
    output = model(dummy_input)
    assert output.shape == (5, 4)


def test_unsupervised_autoencoder_scoring():
    # Instantiate with fallback/untrained weights
    ae = UnsupervisedAutoencoder(input_dim=4, latent_dim=3)

    # Test random input scoring
    dummy_features = np.random.normal(loc=0.0, scale=1.0, size=(4,))
    score = ae.compute_anomaly_score(dummy_features)

    # Verify bounds
    assert 0.0 <= score <= 1.0
    assert isinstance(score, float)


# --- 2. Trilateration Plausibility Tests ---


def test_trilateration_plausibility():
    # SF (37.7749, -122.4194)
    # Oakland (37.8044, -122.2712)
    # Fresno (36.7783, -119.4179)

    # Case A: Insufficient sensors (less than 2 valid) -> Inconclusive
    consistency, reason, evidence = check_trilateration_plausibility(
        aircraft_lat=37.7749, aircraft_lon=-122.4194, sensors=["1"]  # only one sensor
    )
    assert consistency == 1.0
    assert reason == "inconclusive"
    assert "reason" in evidence

    # Case B: Consistent geometry (aircraft at SF, sensors in SF & Oakland) -> Consistent
    consistency, reason, evidence = check_trilateration_plausibility(
        aircraft_lat=37.7749, aircraft_lon=-122.4194, sensors=["1", "2"]
    )
    assert consistency == 1.0
    assert reason == "consistent"
    assert len(evidence["distances_km"]) == 2
    assert all(d < 50.0 for d in evidence["distances_km"])

    # Case C: Inconsistent geometry (aircraft at SF, but Fresno sensor "5" reports it, distance ~270km, plus mock invalid sensor) -> Consistent still (distance under 350km)
    consistency, reason, evidence = check_trilateration_plausibility(
        aircraft_lat=37.7749, aircraft_lon=-122.4194, sensors=["1", "5"]
    )
    assert consistency == 1.0
    assert reason == "consistent"

    # Case D: Inconsistent geometry (aircraft reported far out in Pacific Ocean, but SF & Oakland sensors report it -> distance > 1000km) -> Inconsistent!
    consistency, reason, evidence = check_trilateration_plausibility(
        aircraft_lat=30.0000, aircraft_lon=-130.0000, sensors=["1", "2"]
    )
    assert consistency < 1.0
    assert "physically inconsistent geometry" in reason
    assert evidence["max_distance_km"] > 350.0


# --- 3. Combined Scoring Weight Boundary Tests ---


def test_combined_scoring_boundaries():
    # combine_scores args: rule_flags, ensemble_score, autoencoder_score, trilateration_consistency, threshold

    # Boundary 1: All inputs zero (No rules, zero ML probabilities, full consistency)
    risk, triggered = combine_scores(
        rule_flags=[False, False, False, False],
        ensemble_score=0.0,
        autoencoder_score=0.0,
        trilateration_consistency=1.0,
        threshold=0.7,
    )
    assert risk == 0.0
    assert not triggered

    # Boundary 2: Max components except rules (ensemble=1.0, autoencoder=1.0, inconsistency=1.0)
    # score = 0.40*(0) + 0.30*(1.0) + 0.20*(1.0) + 0.10*(1.0) = 0.60
    risk, triggered = combine_scores(
        rule_flags=[False, False, False, False],
        ensemble_score=1.0,
        autoencoder_score=1.0,
        trilateration_consistency=0.0,  # consistency 0 -> inconsistency 1
        threshold=0.7,
    )
    assert abs(risk - 0.60) < 1e-5
    assert not triggered

    # Boundary 3: Only rules triggered (rule_risk = 1.0)
    # score = 0.40*(1.0) + 0.30*(0) + 0.20*(0) + 0.10*(0) = 0.40
    risk, triggered = combine_scores(
        rule_flags=[True, False, False, False],
        ensemble_score=0.0,
        autoencoder_score=0.0,
        trilateration_consistency=1.0,
        threshold=0.7,
    )
    assert abs(risk - 0.40) < 1e-5
    assert not triggered

    # Boundary 4: Threshold gate boundary (Rules + Ensemble = 0.70)
    # score = 0.40*(1.0) + 0.30*(1.0) + 0.20*(0) + 0.10*(0) = 0.70
    risk, triggered = combine_scores(
        rule_flags=[True, False, False, False],
        ensemble_score=1.0,
        autoencoder_score=0.0,
        trilateration_consistency=1.0,
        threshold=0.7,
    )
    assert abs(risk - 0.70) < 1e-5
    assert triggered

    # Boundary 5: All components maxed (Rules + Ensemble + AE + Inconsistency = 1.0)
    risk, triggered = combine_scores(
        rule_flags=[True, False, False, False],
        ensemble_score=1.0,
        autoencoder_score=1.0,
        trilateration_consistency=0.0,
        threshold=0.7,
    )
    assert abs(risk - 1.0) < 1e-5
    assert triggered
