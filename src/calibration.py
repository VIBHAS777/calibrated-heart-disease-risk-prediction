import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

class ProbabilisticCalibrator:
    """
    Post-hoc probability calibration techniques for deep learning risk predictors:
      1. Platt Scaling (Sigmoidal logistic transform)
      2. Isotonic Regression (Non-parametric isotonic mapping)
    Computes Brier Score and Expected Calibration Error (ECE).
    """
    def __init__(self, method: str = "platt"):
        self.method = method
        if method == "platt":
            self.calibrator = LogisticRegression()
        elif method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
        else:
            raise ValueError(f"Unknown calibration method: {method}")

    def fit(self, uncalibrated_probs: np.ndarray, y_true: np.ndarray):
        if self.method == "platt":
            self.calibrator.fit(uncalibrated_probs.reshape(-1, 1), y_true)
        else:
            self.calibrator.fit(uncalibrated_probs, y_true)

    def calibrate(self, uncalibrated_probs: np.ndarray) -> np.ndarray:
        if self.method == "platt":
            return self.calibrator.predict_proba(uncalibrated_probs.reshape(-1, 1))[:, 1]
        else:
            return self.calibrator.predict(uncalibrated_probs)

    @staticmethod
    def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        return float(np.mean((y_prob - y_true) ** 2))
