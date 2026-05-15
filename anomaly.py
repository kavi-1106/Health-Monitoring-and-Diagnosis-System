"""
routers/anomaly.py
POST /anomaly/{patient_id}/analyze   — run ML anomaly detection on stored data
POST /anomaly/check                  — rule-based check on a single reading
GET  /anomaly/{patient_id}/summary   — anomaly statistics
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from schemas.health import VitalsInput, AnomalyResult, AnomalyBatchResponse
from services.data_store import store
from services.detector import (rule_check, extract_features,
                                train_detector, get_detector)

router = APIRouter(prefix="/anomaly", tags=["Anomaly Detection"])


@router.post("/check", response_model=AnomalyResult,
             summary="Real-time rule-based check on a single vitals reading")
def check_single(body: VitalsInput):
    """
    Instantly checks one reading against clinical thresholds.
    No ML model required — returns severity + alert list.
    """
    reading = body.model_dump()
    severity, alerts = rule_check(reading)
    return AnomalyResult(
        timestamp=body.timestamp,
        is_anomaly=(severity != 'normal'),
        confidence=1.0 if severity == 'critical' else (0.7 if severity == 'warning' else 0.0),
        severity=severity,
        alerts=alerts,
    )


@router.post("/{patient_id}/analyze", response_model=AnomalyBatchResponse,
             summary="Train ML detector and analyze patient history")
def analyze_patient(
    patient_id: str,
    window: int = Query(10, ge=5, le=60,
                        description="Feature extraction window size (minutes)")
):
    """
    Runs the full ML pipeline on all stored vitals for a patient:
    1. Extracts time/frequency/wavelet features per window
    2. Trains Isolation Forest (+ Random Forest if anomaly labels exist)
    3. Returns per-window predictions with confidence scores
    """
    df = store.get_dataframe(patient_id)
    if df.empty:
        raise HTTPException(404, f"No vitals data for patient '{patient_id}'")
    if len(df) < window:
        raise HTTPException(422, f"Need at least {window} readings (got {len(df)})")

    detector, features = train_detector(df, window=window)
    preds = detector.predict(features)

    results = []
    for _, row in preds.iterrows():
        is_anom = bool(row['is_anomaly'])
        conf    = float(row['confidence'])
        sev     = 'normal'
        if is_anom:
            sev = 'critical' if conf > 0.85 else 'warning'
        results.append(AnomalyResult(
            timestamp=row.get('timestamp'),
            is_anomaly=is_anom,
            confidence=conf,
            severity=sev,
            alerts=[f"ML anomaly detected (confidence {conf*100:.0f}%)"] if is_anom else [],
            iso_score=float(row.get('iso_conf', 0)),
            rf_score=float(row.get('rf_conf', 0)) if 'rf_conf' in row else None,
        ))

    n_anom = sum(1 for r in results if r.is_anomaly)
    return AnomalyBatchResponse(
        patient_id=patient_id,
        total_windows=len(results),
        anomalies_found=n_anom,
        anomaly_rate=round(n_anom / max(len(results), 1), 4),
        results=results,
    )


@router.get("/{patient_id}/summary",
            summary="Anomaly statistics for a patient")
def anomaly_summary(
    patient_id: str,
    last_n: Optional[int] = Query(None, ge=1,
                                   description="Analyze only last N readings")
):
    """Quick summary: how many readings are flagged by clinical rules."""
    readings = store.get_readings(patient_id, last_n=last_n)
    if not readings:
        raise HTTPException(404, f"No vitals for patient '{patient_id}'")

    total    = len(readings)
    warnings = sum(1 for r in readings if rule_check(r)[0] == 'warning')
    critical = sum(1 for r in readings if rule_check(r)[0] == 'critical')

    return {
        "patient_id":       patient_id,
        "total_readings":   total,
        "normal":           total - warnings - critical,
        "warnings":         warnings,
        "critical":         critical,
        "anomaly_rate_pct": round((warnings + critical) / total * 100, 2),
    }
