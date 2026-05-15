"""
data/sensor_simulator.py
Simulates wearable sensor data: ECG, SpO2, temperature, accelerometer.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_ecg(duration_sec=60, fs=250, anomaly=False):
    """Generate synthetic ECG waveform."""
    t = np.linspace(0, duration_sec, duration_sec * fs)
    ecg = np.zeros_like(t)

    bpm = 72 if not anomaly else np.random.choice([45, 110, 130])
    rr_interval = 60.0 / bpm

    beat_times = np.arange(0, duration_sec, rr_interval)

    for bt in beat_times:
        idx = np.searchsorted(t, bt)
        window = np.arange(idx, min(idx + int(0.6 * fs), len(t)))
        local_t = (t[window] - bt)

        # P wave
        ecg[window] += 0.15 * np.exp(-((local_t - 0.10) ** 2) / (2 * 0.012 ** 2))
        # Q wave
        ecg[window] -= 0.05 * np.exp(-((local_t - 0.18) ** 2) / (2 * 0.005 ** 2))
        # R wave
        ecg[window] += 1.20 * np.exp(-((local_t - 0.20) ** 2) / (2 * 0.006 ** 2))
        # S wave
        ecg[window] -= 0.25 * np.exp(-((local_t - 0.23) ** 2) / (2 * 0.006 ** 2))
        # T wave
        t_offset = 0.36 if not anomaly else 0.42  # prolonged QTc if anomaly
        ecg[window] += 0.35 * np.exp(-((local_t - t_offset) ** 2) / (2 * 0.030 ** 2))

    noise = np.random.normal(0, 0.02, len(t))
    ecg += noise
    return t, ecg


def generate_vitals(n_samples=1440, anomaly_rate=0.05):
    """
    Generate 24h of minute-resolution vitals.
    Returns a DataFrame with heart_rate, spo2, temperature, blood_pressure_sys,
    blood_pressure_dia, resp_rate, steps, hrv.
    """
    np.random.seed(42)
    timestamps = [datetime(2024, 1, 15, 0, 0) + timedelta(minutes=i)
                  for i in range(n_samples)]

    # Base circadian rhythm
    hour = np.array([ts.hour + ts.minute / 60 for ts in timestamps])
    circadian = np.sin(2 * np.pi * (hour - 6) / 24)

    # Heart rate: 55-85 bpm, lower at night
    hr_base = 68 - 8 * circadian
    hr = hr_base + np.random.normal(0, 3, n_samples)

    # SpO2: 94-99%
    spo2 = 97 + np.random.normal(0, 0.8, n_samples)
    spo2 = np.clip(spo2, 85, 100)

    # Temperature: 36.2-37.5 C, peaks late afternoon
    temp_base = 36.8 + 0.4 * np.sin(2 * np.pi * (hour - 14) / 24)
    temp = temp_base + np.random.normal(0, 0.1, n_samples)

    # Blood pressure
    bp_sys_base = 118 + 10 * (1 - circadian)
    bp_sys = bp_sys_base + np.random.normal(0, 5, n_samples)
    bp_dia = bp_sys * 0.62 + np.random.normal(0, 3, n_samples)

    # Respiratory rate: 12-20 bpm
    resp = 15 + np.random.normal(0, 1.5, n_samples)
    resp = np.clip(resp, 8, 30)

    # Steps (active during day)
    activity = np.where((hour >= 7) & (hour <= 21), 1, 0)
    steps_per_min = activity * np.abs(np.random.normal(5, 3, n_samples))

    # HRV (ms) - inverse relationship with stress
    hrv = 45 - 10 * (1 - circadian) + np.random.normal(0, 8, n_samples)
    hrv = np.clip(hrv, 10, 100)

    # Inject anomalies
    n_anomalies = int(n_samples * anomaly_rate)
    anomaly_idx = np.random.choice(n_samples, n_anomalies, replace=False)
    anomaly_labels = np.zeros(n_samples, dtype=int)

    for idx in anomaly_idx:
        kind = np.random.choice(['hypertension', 'bradycardia', 'tachycardia',
                                  'hypoxia', 'fever'])
        anomaly_labels[idx] = 1
        if kind == 'hypertension':
            bp_sys[idx] += np.random.uniform(30, 60)
            bp_dia[idx] += np.random.uniform(15, 30)
        elif kind == 'bradycardia':
            hr[idx] = np.random.uniform(35, 48)
        elif kind == 'tachycardia':
            hr[idx] = np.random.uniform(115, 160)
        elif kind == 'hypoxia':
            spo2[idx] = np.random.uniform(82, 91)
        elif kind == 'fever':
            temp[idx] = np.random.uniform(38.2, 40.1)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'heart_rate': np.round(hr, 1),
        'spo2': np.round(spo2, 1),
        'temperature': np.round(temp, 2),
        'bp_systolic': np.round(bp_sys, 1),
        'bp_diastolic': np.round(bp_dia, 1),
        'resp_rate': np.round(resp, 1),
        'steps_per_min': np.round(steps_per_min, 1),
        'hrv_ms': np.round(hrv, 1),
        'anomaly': anomaly_labels
    })
    return df


if __name__ == '__main__':
    df = generate_vitals()
    df.to_csv('data/vitals.csv', index=False)
    print(f"Generated {len(df)} samples with {df['anomaly'].sum()} anomalies.")
    print(df.head())
