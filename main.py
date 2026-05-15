"""
main.py
Entry point for the Health Monitoring & Diagnosis System.
Run: python main.py
"""

import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data.sensor_simulator import generate_vitals
from utils.feature_extraction import extract_window_features
from models.anomaly_detector import HealthAnomalyDetector, rule_based_check
from reports.report_generator import generate_pdf_report, plot_vitals_summary


def run_pipeline(patient_info: dict, n_samples: int = 1440,
                 window: int = 10, anomaly_rate: float = 0.05,
                 out_dir: str = 'reports'):

    print("=" * 55)
    print("  HEALTH MONITORING & DIAGNOSIS SYSTEM")
    print("=" * 55)

    # 1. Simulate / load sensor data
    print("\n[1/5] Generating sensor data...")
    df = generate_vitals(n_samples=n_samples, anomaly_rate=anomaly_rate)
    df.to_csv(os.path.join(out_dir, 'raw_vitals.csv'), index=False)
    print(f"     {len(df)} samples generated, {df['anomaly'].sum()} true anomalies injected.")

    # 2. Rule-based real-time checks (last reading)
    print("\n[2/5] Running rule-based anomaly checks (last 5 readings)...")
    for _, row in df.tail(5).iterrows():
        result = rule_based_check(row)
        ts = row['timestamp'].strftime('%H:%M')
        if result['severity'] != 'normal':
            for alert in result['alerts']:
                print(f"     [{ts}] {alert}")
        else:
            print(f"     [{ts}] All vitals normal.")

    # 3. Feature extraction
    print(f"\n[3/5] Extracting ML features (window={window} min)...")
    features = extract_window_features(df, window=window)
    features.to_csv(os.path.join(out_dir, 'features.csv'), index=False)
    print(f"     {len(features)} feature windows, {features.shape[1]-2} features each.")

    # 4. Train & predict
    print("\n[4/5] Training anomaly detector...")
    detector = HealthAnomalyDetector(contamination=anomaly_rate)
    detector.fit(features)
    predictions = detector.predict(features)
    predictions.to_csv(os.path.join(out_dir, 'predictions.csv'), index=False)
    detector.save(os.path.join('models', 'detector.pkl'))

    n_flagged = predictions['final_anomaly'].sum()
    avg_conf  = predictions.loc[predictions['final_anomaly'] == 1, 'confidence'].mean()
    print(f"     Flagged {n_flagged} windows | Avg confidence: {avg_conf*100:.0f}%")

    # 5. Generate report
    print("\n[5/5] Generating diagnostic report...")
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, f"report_{patient_info['id']}.pdf")
    generate_pdf_report(patient_info, df, predictions, out_path=pdf_path)
    plot_vitals_summary(df, out_path=os.path.join(out_dir, 'vitals_chart.png'))

    print("\n" + "=" * 55)
    print("  PIPELINE COMPLETE")
    print(f"  Report     : {pdf_path}")
    print(f"  Raw data   : {out_dir}/raw_vitals.csv")
    print(f"  Predictions: {out_dir}/predictions.csv")
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(
        description='Health Monitoring & Diagnosis System'
    )
    parser.add_argument('--patient-id',   default='PT-001', help='Patient ID')
    parser.add_argument('--name',         default='Demo Patient', help='Patient name')
    parser.add_argument('--age',          type=int, default=45, help='Patient age')
    parser.add_argument('--gender',       default='Female', help='Patient gender')
    parser.add_argument('--samples',      type=int, default=1440,
                        help='Number of 1-min samples (1440 = 24h)')
    parser.add_argument('--window',       type=int, default=10,
                        help='Feature window size (minutes)')
    parser.add_argument('--anomaly-rate', type=float, default=0.05,
                        help='Injected anomaly rate 0–1')
    parser.add_argument('--out-dir',      default='reports',
                        help='Output directory')
    args = parser.parse_args()

    patient_info = {
        'id':     args.patient_id,
        'name':   args.name,
        'age':    args.age,
        'gender': args.gender,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    run_pipeline(
        patient_info=patient_info,
        n_samples=args.samples,
        window=args.window,
        anomaly_rate=args.anomaly_rate,
        out_dir=args.out_dir,
    )


if __name__ == '__main__':
    main()
