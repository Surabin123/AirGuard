import os
import joblib
import numpy as np
import shap
from typing import Dict, Any, Tuple

# --- Feature Names ---
FEATURE_NAMES = [
    "speed_variance",
    "heading_variance",
    "altitude_rate_variance",
    "time_since_last_update",
    "rule_position_jump",
    "rule_duplicate_icao",
    "rule_climb_rate",
    "rule_alt_vel_mismatch",
]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "ensemble_model.joblib")


class TrustScoringEnsemble:
    def __init__(self):
        self.model = None
        self.explainer = None
        self.load_model()

    def load_model(self) -> None:
        """Attempt to load trained ensemble model and initialize SHAP explainer."""
        if os.path.exists(MODEL_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                # Initialize TreeExplainer using the RandomForest sub-estimator
                # in the soft-voting ensemble for reliability.
                rf_estimator = self.model.named_estimators_["rf"]
                self.explainer = shap.TreeExplainer(rf_estimator)
            except Exception:
                # If loading fails (e.g. file corrupted/unsupported), log and proceed
                self.model = None
                self.explainer = None

    def predict_anomaly(
        self, feature_vector: np.ndarray
    ) -> Tuple[float, Dict[str, Any]]:
        """Predict anomaly probability and generate top-3 SHAP feature contributions.

        Returns:
            anomaly_probability: float (0.0 to 1.0)
            shap_explanation: dict containing top 3 features and base value.
        """
        if self.model is None:
            # Fallback if model is not trained/loaded: use sum of rule flags
            rule_flags_sum = sum(feature_vector[4:])
            prob = min(1.0, rule_flags_sum * 0.4)
            fallback_explanation = {
                "top_features": [
                    {"feature": FEATURE_NAMES[i], "value": float(feature_vector[i])}
                    for i in range(4, 8)
                    if feature_vector[i] > 0
                ][:3],
                "base_value": 0.02,
                "note": "fallback_no_trained_model",
            }
            return prob, fallback_explanation

        # Ensure correct shape
        X = feature_vector.reshape(1, -1)

        # Get probability of positive class (anomaly, index 1)
        prob = float(self.model.predict_proba(X)[0][1])

        # Compute SHAP values
        shap_explanation = {"top_features": [], "base_value": 0.0}
        if self.explainer is not None:
            try:
                # Get SHAP values for class 1 (anomaly)
                raw_shap = self.explainer.shap_values(X)

                # Handle SHAP output format variations
                # In binary classification, shap_values can be a list of two arrays [class_0, class_1]
                # or a single array of shape (1, 8, 2)
                if isinstance(raw_shap, list):
                    shap_vals = raw_shap[1][0]
                elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
                    shap_vals = raw_shap[0, :, 1]
                else:
                    # Fallback single array output
                    shap_vals = raw_shap[0] if raw_shap.ndim == 2 else raw_shap

                # Pair with feature names and absolute sort
                paired = []
                for idx, name in enumerate(FEATURE_NAMES):
                    paired.append({"feature": name, "value": float(shap_vals[idx])})

                # Sort by absolute SHAP value descending
                paired.sort(key=lambda x: abs(x["value"]), reverse=True)

                # Take top-3 contributors
                shap_explanation = {
                    "top_features": paired[:3],
                    "base_value": (
                        float(self.explainer.expected_value[1])
                        if isinstance(self.explainer.expected_value, (list, np.ndarray))
                        else float(self.explainer.expected_value)
                    ),
                }
            except Exception as e:
                shap_explanation = {
                    "top_features": [],
                    "base_value": 0.0,
                    "error": str(e),
                }

        return prob, shap_explanation
