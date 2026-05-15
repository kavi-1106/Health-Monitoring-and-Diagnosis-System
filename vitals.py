"""
routers/vitals.py
POST /vitals/{patient_id}        — submit a single reading
POST /vitals/{patient_id}/batch  — submit bulk readings
GET  /vitals/{patient_id}        — fetch readings (with optional limit)
GET  /vitals/{patient_id}/latest — fetch the most recent reading
DELETE /vitals/{patient_id}      — clear all readings
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime

from schemas.health import VitalsInput, VitalsResponse, VitalsBatchInput
from services.data_store import store
from services.detector import rule_check

router = APIRouter(prefix="/vitals", tags=["Vitals"])


def _enrich(reading: dict) -> dict:
    severity, alerts = rule_check(reading)
    return {**reading, 'rule_severity': severity, 'rule_alerts': alerts}


@router.post("/{patient_id}", response_model=VitalsResponse, status_code=201,
             summary="Submit a single vitals reading")
def submit_vitals(patient_id: str, body: VitalsInput):
    """
    Submit one vitals reading for a patient.
    Returns the stored reading with real-time rule-based alerts.
    """
    reading = body.model_dump()
    reading['timestamp'] = reading['timestamp'] or datetime.utcnow()
    reading['patient_id'] = patient_id
    store.add_reading(patient_id, reading)
    return _enrich(reading)


@router.post("/{patient_id}/batch", status_code=201,
             summary="Submit a batch of vitals readings")
def submit_batch(patient_id: str, body: VitalsBatchInput):
    """Submit multiple readings at once (e.g. from a wearable sync)."""
    added = 0
    for r in body.readings:
        reading = r.model_dump()
        reading['timestamp'] = reading['timestamp'] or datetime.utcnow()
        reading['patient_id'] = patient_id
        store.add_reading(patient_id, reading)
        added += 1
    return {"patient_id": patient_id, "readings_added": added}


@router.get("/{patient_id}", response_model=List[VitalsResponse],
            summary="Get vitals history")
def get_vitals(
    patient_id: str,
    limit: Optional[int] = Query(None, ge=1, le=10080,
                                  description="Max readings to return (default: all)")
):
    """Retrieve stored vitals for a patient, newest last."""
    readings = store.get_readings(patient_id, last_n=limit)
    if not readings:
        raise HTTPException(404, f"No vitals found for patient '{patient_id}'")
    return [_enrich(r) for r in readings]


@router.get("/{patient_id}/latest", response_model=VitalsResponse,
            summary="Get the most recent vitals reading")
def get_latest(patient_id: str):
    readings = store.get_readings(patient_id, last_n=1)
    if not readings:
        raise HTTPException(404, f"No vitals found for patient '{patient_id}'")
    return _enrich(readings[0])


@router.delete("/{patient_id}", summary="Clear all readings for a patient")
def clear_vitals(patient_id: str):
    store.clear(patient_id)
    return {"message": f"All readings cleared for patient '{patient_id}'"}
