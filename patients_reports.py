"""
routers/patients.py
POST /patients          — register patient
GET  /patients          — list all patients
GET  /patients/{id}     — get patient info
GET  /patients/{id}/score — health score

routers/reports.py
GET  /reports/{id}/pdf  — download PDF report
GET  /reports/{id}/chart — download vitals chart PNG
GET  /reports/{id}/score — health score JSON
GET  /reports/{id}/simulate — simulate + auto-train + return score
"""

# ── patients.py ───────────────────────────────────────────────────────────────
from fastapi import APIRouter, HTTPException
from schemas.health import PatientInfo, HealthScoreResponse, SimulateRequest
from services.data_store import store, simulate_vitals
from services.detector import train_detector
from services.report_service import compute_health_score, generate_pdf, generate_vitals_chart
from fastapi.responses import Response

patients_router = APIRouter(prefix="/patients", tags=["Patients"])


@patients_router.post("", status_code=201, summary="Register a new patient")
def register_patient(body: PatientInfo):
    store.register_patient(body.patient_id, body.model_dump())
    return {"message": f"Patient '{body.patient_id}' registered.", **body.model_dump()}


@patients_router.get("", summary="List all registered patients")
def list_patients():
    return {"patients": store.list_patients()}


@patients_router.get("/{patient_id}", summary="Get patient info")
def get_patient(patient_id: str):
    info = store.get_patient(patient_id)
    if not info:
        raise HTTPException(404, f"Patient '{patient_id}' not found")
    return info


@patients_router.get("/{patient_id}/score",
                      response_model=HealthScoreResponse,
                      summary="Compute health score from stored vitals")
def health_score(patient_id: str):
    df = store.get_dataframe(patient_id)
    if df.empty:
        raise HTTPException(404, f"No vitals for patient '{patient_id}'")
    score = compute_health_score(df)
    interp = ("Excellent" if score['total'] >= 90 else
              "Good"      if score['total'] >= 80 else
              "Fair"      if score['total'] >= 70 else
              "Poor"      if score['total'] >= 60 else "Critical")
    return HealthScoreResponse(
        patient_id=patient_id,
        total_score=score['total'],
        grade=score['grade'],
        breakdown=score['breakdown'],
        interpretation=interp,
    )


# ── reports.py ────────────────────────────────────────────────────────────────
reports_router = APIRouter(prefix="/reports", tags=["Reports"])


@reports_router.get("/{patient_id}/pdf",
                     summary="Download PDF diagnostic report")
def download_pdf(patient_id: str):
    df   = store.get_dataframe(patient_id)
    info = store.get_patient(patient_id) or {'patient_id': patient_id,
                                              'name': 'Unknown', 'age': 'N/A',
                                              'gender': 'N/A'}
    if df.empty:
        raise HTTPException(404, f"No vitals for patient '{patient_id}'")
    pdf_bytes = generate_pdf(info, df)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=report_{patient_id}.pdf"}
    )


@reports_router.get("/{patient_id}/chart",
                     summary="Download 24h vitals chart as PNG")
def download_chart(patient_id: str):
    df = store.get_dataframe(patient_id)
    if df.empty:
        raise HTTPException(404, f"No vitals for patient '{patient_id}'")
    png = generate_vitals_chart(df)
    return Response(
        content=png,
        media_type="image/png",
        headers={"Content-Disposition": f"inline; filename=chart_{patient_id}.png"}
    )


@reports_router.post("/simulate",
                      summary="Simulate 24h data, train model, return full report")
def simulate_and_report(body: SimulateRequest):
    """
    One-shot endpoint: generates synthetic data, trains the ML detector,
    computes the health score, and returns a JSON summary.
    Use GET /reports/{id}/pdf afterwards for the PDF.
    """
    df = simulate_vitals(n_samples=body.n_samples, anomaly_rate=body.anomaly_rate)
    for _, row in df.iterrows():
        store.add_reading(body.patient_id, row.to_dict())

    detector, features = train_detector(df, window=10)
    preds  = detector.predict(features)
    score  = compute_health_score(df, preds)
    n_anom = int(preds['is_anomaly'].sum())

    return {
        "patient_id":       body.patient_id,
        "samples_generated": len(df),
        "anomalies_injected": int(df['anomaly'].sum()),
        "anomalies_detected": n_anom,
        "health_score":      score['total'],
        "grade":             score['grade'],
        "breakdown":         score['breakdown'],
        "pdf_url":           f"/reports/{body.patient_id}/pdf",
        "chart_url":         f"/reports/{body.patient_id}/chart",
    }
