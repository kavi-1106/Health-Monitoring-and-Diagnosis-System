"""
models/anomaly_detector.py
Anomaly detection pipeline:
  1. Rule-based clinical thresholds (fast, interpretable)
  2. Isolation Forest on extracted features (ML)
  3. Combined scoring with confidence estimation
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# ── Clinical thresholds ────────────────────────────────────────────────────────

THRESHOLDS = {
    'heart_rate':    {'low': 50,  'high': 100, 'critical_low': 40, 'critical_high': 130},
    'spo2':          {'low': 94,  'high': 100, 'critical_low': 90, 'critical_high': None},
    'temperature':   {'low': 36.0,'high': 37.5,'critical_low': 35.0,'critical_high': 38.5},
    'bp_systolic':   {'low': 90,  'high': 130, 'critical_low': 80, 'critical_high': 180},
    'bp_diastolic':  {'low': 60,  'high': 85,  'critical_low': 50, 'critical_high': 120},
    'resp_rate':     {'low': 12,  'high': 20,  'critical_low': 8,  'critical_high': 25},
    'hrv_ms':        {'low': 20,  'high': 80,  'critical_low': 10, 'critical_high': None},
}


def rule_based_check(row: pd.Series) -> dict:
    """Return alert level and details for a single vitals row."""
    alerts = []
    severity = 'normal'

    for metric, limits in THRESHOLDS.items():
        if metric not in row:
            continue
        val = row[metric]
        name = metric.replace('_', ' ').title()

        if limits['critical_low'] and val < limits['critical_low']:
            alerts.append(f"CRITICAL: {name} = {val:.1f} (below {limits['critical_low']})")
            severity = 'critical'
        elif limits['critical_high'] and val > limits['critical_high']:
            alerts.append(f"CRITICAL: {name} = {val:.1f} (above {limits['critical_high']})")
            severity = 'critical'
        elif val < limits['low']:
            alerts.append(f"WARNING: {name} = {val:.1f} (below {limits['low']})")
            if severity != 'critical':
                severity = 'warning'
        elif limits['high'] and val > limits['high']:
            alerts.append(f"WARNING: {name} = {val:.1f} (above {limits['high']})")
            if severity != 'critical':
                severity = 'warning'

    return {'severity': severity, 'alerts': alerts}


# ── ML-based detector ──────────────────────────────────────────────────────────

class HealthAnomalyDetector:
    """
    Two-stage detector:
      Stage 1 — Isolation Forest (unsupervised, no labels needed)
      Stage 2 — Random Forest classifier (if labeled data available)
    """

    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.iso_forest = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        self.rf_classifier = RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.is_trained = False
        self.has_classifier = False

    def _get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        exclude = {'timestamp', 'anomaly'}
        cols = [c for c in df.columns if c not in exclude]
        self.feature_cols = cols
        return df[cols].fillna(0).values

    def fit(self, df: pd.DataFrame):
        """Train on feature DataFrame. Uses RF if 'anomaly' labels exist."""
        X = self._get_feature_matrix(df)
        X_scaled = self.scaler.fit_transform(X)

        # Isolation Forest always trained
        self.iso_forest.fit(X_scaled)
        print(f"[IsolationForest] Trained on {len(X)} windows.")

        # Random Forest if labels present
        if 'anomaly' in df.columns and df['anomaly'].nunique() > 1:
            y = df['anomaly'].values
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            self.rf_classifier.fit(X_tr, y_tr)
            y_pred = self.rf_classifier.predict(X_te)
            print("\n[RandomForest] Classification report:")
            print(classification_report(y_te, y_pred,
                  target_names=['Normal', 'Anomaly']))
            self.has_classifier = True

        self.is_trained = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return predictions with confidence scores."""
        if not self.is_trained:
            raise RuntimeError("Call .fit() before .predict()")

        cols = [c for c in self.feature_cols if c in df.columns]
        X = df[cols].fillna(0).values
        X_scaled = self.scaler.transform(X)

        # Isolation Forest scores (lower = more anomalous)
        iso_scores = self.iso_forest.score_samples(X_scaled)
        iso_pred = self.iso_forest.predict(X_scaled)  # -1 anomaly, 1 normal
        iso_anomaly = (iso_pred == -1).astype(int)

        # Normalize score to 0-1 confidence
        s_min, s_max = iso_scores.min(), iso_scores.max()
        if s_max > s_min:
            iso_confidence = 1 - (iso_scores - s_min) / (s_max - s_min)
        else:
            iso_confidence = np.zeros(len(iso_scores))

        result = df[['timestamp']].copy() if 'timestamp' in df.columns else pd.DataFrame()
        result['iso_anomaly'] = iso_anomaly
        result['iso_confidence'] = np.round(iso_confidence, 3)

        if self.has_classifier:
            rf_pred = self.rf_classifier.predict(X_scaled)
            rf_proba = self.rf_classifier.predict_proba(X_scaled)[:, 1]
            result['rf_anomaly'] = rf_pred
            result['rf_confidence'] = np.round(rf_proba, 3)
            # Ensemble: anomaly if either model flags it
            result['final_anomaly'] = ((iso_anomaly == 1) | (rf_pred == 1)).astype(int)
            result['confidence'] = np.round(
                0.4 * iso_confidence + 0.6 * rf_proba, 3
            )
        else:
            result['final_anomaly'] = iso_anomaly
            result['confidence'] = iso_confidence

        return result

    def save(self, path: str = 'models/detector.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {path}")

    @staticmethod
    def load(path: str = 'models/detector.pkl'):
        with open(path, 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from data.sensor_simulator import generate_vitals
    from utils.feature_extraction import extract_window_features

    print("Generating data...")
    df = generate_vitals(n_samples=1440)
    print("Extracting features...")
    features = extract_window_features(df, window=10)

    detector = HealthAnomalyDetector(contamination=0.05)
    detector.fit(features)
    predictions = detector.predict(features)
    detector.save()

    n_flagged = predictions['final_anomaly'].sum()
    print(f"\nFlagged {n_flagged} / {len(predictions)} windows as anomalous.")
