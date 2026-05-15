"""
reports/report_generator.py
Generates a structured PDF diagnostic report for a patient session.
Falls back to a text report if ReportLab is not installed.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, Image, HRFlowable)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB = True
except ImportError:
    REPORTLAB = False
    print("[WARN] ReportLab not found — text report will be generated instead.")


# ── Chart generation ──────────────────────────────────────────────────────────

def plot_vitals_summary(df: pd.DataFrame, out_path: str = 'reports/vitals_plot.png'):
    """Generate a 4-panel vitals chart."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig = plt.figure(figsize=(14, 8), facecolor='#0A0F1A')
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    panels = [
        ('heart_rate',  'Heart Rate (bpm)',      '#00E5A0', (40, 160)),
        ('spo2',        'SpO₂ (%)',               '#4DA8FF', (80, 102)),
        ('bp_systolic', 'Systolic BP (mmHg)',     '#FF4D6D', (70, 210)),
        ('temperature', 'Body Temperature (°C)',  '#FFB830', (34, 41)),
    ]

    for i, (col, title, color, ylim) in enumerate(panels):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        ax.set_facecolor('#111827')
        ax.tick_params(colors='#6B7A99', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#1E293B')

        x = df['timestamp'] if 'timestamp' in df else range(len(df))
        ax.plot(x, df[col], color=color, linewidth=0.8, alpha=0.9)

        # Shade anomalies
        if 'anomaly' in df.columns:
            for idx in df.index[df['anomaly'] == 1]:
                ax.axvline(x=df.loc[idx, 'timestamp'] if 'timestamp' in df else idx,
                           color='#FF4D6D', alpha=0.3, linewidth=0.5)

        ax.set_title(title, color='#E8EDF5', fontsize=9, pad=6, fontweight='bold')
        ax.set_ylim(ylim)
        ax.grid(True, color='#1E293B', linewidth=0.5, linestyle='--')
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

    fig.suptitle('24-Hour Vitals Overview', color='#00E5A0',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='#0A0F1A', edgecolor='none')
    plt.close()
    return out_path


# ── Health score ──────────────────────────────────────────────────────────────

def compute_health_score(df: pd.DataFrame, predictions: pd.DataFrame = None) -> dict:
    """Compute a composite health score 0-100."""
    scores = {}

    def score_metric(val, low, high, weight=1.0):
        if low <= val <= high:
            return 100 * weight
        dist = min(abs(val - low), abs(val - high))
        span = (high - low) / 2
        return max(0, (1 - dist / span) * 100) * weight

    last = df.iloc[-60:] if len(df) > 60 else df  # last hour
    avg = last.mean(numeric_only=True)

    scores['heart_rate']   = score_metric(avg.get('heart_rate', 70),   55, 90,  0.20)
    scores['spo2']         = score_metric(avg.get('spo2', 97),         95, 100, 0.25)
    scores['temperature']  = score_metric(avg.get('temperature', 37),  36.1, 37.2, 0.15)
    scores['bp_systolic']  = score_metric(avg.get('bp_systolic', 120), 90, 130, 0.20)
    scores['hrv']          = score_metric(avg.get('hrv_ms', 45),       30, 80,  0.10)
    scores['resp_rate']    = score_metric(avg.get('resp_rate', 15),    12, 20,  0.10)

    total = sum(scores.values())
    if predictions is not None and 'final_anomaly' in predictions.columns:
        anomaly_rate = predictions['final_anomaly'].mean()
        penalty = anomaly_rate * 20
        total = max(0, total - penalty)

    return {'total': round(total, 1), 'breakdown': {k: round(v, 1) for k, v in scores.items()}}


# ── Text fallback ─────────────────────────────────────────────────────────────

def generate_text_report(patient_info: dict, df: pd.DataFrame,
                          predictions: pd.DataFrame, out_path: str):
    """Plain-text report when ReportLab is unavailable."""
    score = compute_health_score(df, predictions)
    lines = [
        "=" * 60,
        "    HEALTH MONITORING DIAGNOSTIC REPORT",
        "=" * 60,
        f"Patient ID   : {patient_info.get('id', 'UNKNOWN')}",
        f"Name         : {patient_info.get('name', 'N/A')}",
        f"Age / Gender : {patient_info.get('age', 'N/A')} / {patient_info.get('gender', 'N/A')}",
        f"Report Date  : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Data Period  : {df['timestamp'].min()} — {df['timestamp'].max()}",
        "",
        f"OVERALL HEALTH SCORE: {score['total']} / 100",
        "",
        "SCORE BREAKDOWN:",
    ]
    for k, v in score['breakdown'].items():
        lines.append(f"  {k.replace('_', ' ').title():<20} {v:.1f}")

    avg = df.mean(numeric_only=True)
    lines += [
        "",
        "AVERAGE VITALS (24h):",
        f"  Heart Rate   : {avg.get('heart_rate', 0):.1f} bpm",
        f"  SpO2         : {avg.get('spo2', 0):.1f} %",
        f"  Temperature  : {avg.get('temperature', 0):.2f} °C",
        f"  BP Systolic  : {avg.get('bp_systolic', 0):.1f} mmHg",
        f"  BP Diastolic : {avg.get('bp_diastolic', 0):.1f} mmHg",
        f"  HRV          : {avg.get('hrv_ms', 0):.1f} ms",
    ]

    if predictions is not None:
        n_anom = predictions['final_anomaly'].sum()
        lines += [
            "",
            f"ML ANOMALY DETECTION: {n_anom} windows flagged",
            f"Anomaly rate: {predictions['final_anomaly'].mean()*100:.1f}%",
        ]

    lines += ["", "=" * 60, "End of Report", "=" * 60]
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Text report saved: {out_path}")


# ── PDF report ────────────────────────────────────────────────────────────────

def generate_pdf_report(patient_info: dict, df: pd.DataFrame,
                         predictions: pd.DataFrame = None,
                         out_path: str = 'reports/diagnostic_report.pdf'):
    if not REPORTLAB:
        generate_text_report(patient_info, df, predictions,
                              out_path.replace('.pdf', '.txt'))
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    score = compute_health_score(df, predictions)

    # Generate chart
    chart_path = plot_vitals_summary(df)

    doc = SimpleDocTemplate(out_path, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()

    DARK  = colors.HexColor('#0A0F1A')
    GREEN = colors.HexColor('#00E5A0')
    RED   = colors.HexColor('#FF4D6D')
    AMBER = colors.HexColor('#FFB830')
    BLUE  = colors.HexColor('#4DA8FF')
    GRAY  = colors.HexColor('#6B7A99')

    title_style = ParagraphStyle('Title', parent=styles['Title'],
                                  fontSize=20, textColor=DARK, spaceAfter=4,
                                  alignment=TA_CENTER)
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'],
                               fontSize=13, textColor=DARK, spaceBefore=12, spaceAfter=4)
    body_style = ParagraphStyle('Body', parent=styles['Normal'],
                                 fontSize=10, textColor=colors.HexColor('#334155'),
                                 spaceAfter=4, leading=16)
    label_style = ParagraphStyle('Label', parent=body_style,
                                  textColor=GRAY, fontSize=9)

    story = []

    # Header
    story.append(Paragraph("Health Monitoring Diagnostic Report", title_style))
    story.append(HRFlowable(width='100%', thickness=2, color=GREEN, spaceAfter=10))

    # Patient info table
    info_data = [
        ['Patient ID', patient_info.get('id', 'N/A'),
         'Report Date', datetime.now().strftime('%d %b %Y %H:%M')],
        ['Name', patient_info.get('name', 'N/A'),
         'Age / Gender', f"{patient_info.get('age','N/A')} / {patient_info.get('gender','N/A')}"],
        ['Data Range',
         f"{df['timestamp'].min().strftime('%H:%M')} — {df['timestamp'].max().strftime('%H:%M')}",
         'Total Records', str(len(df))],
    ]
    info_table = Table(info_data, colWidths=[3*cm, 5*cm, 3.5*cm, 5*cm])
    info_table.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), GRAY),
        ('TEXTCOLOR', (2, 0), (2, -1), GRAY),
        ('TEXTCOLOR', (1, 0), (1, -1), DARK),
        ('TEXTCOLOR', (3, 0), (3, -1), DARK),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#E2E8F0')),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1),
         [colors.HexColor('#F8FAFC'), colors.HexColor('#EFF6FF')]),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.5*cm))

    # Health score
    story.append(Paragraph("Overall Health Score", h2_style))
    score_color = GREEN if score['total'] >= 80 else (AMBER if score['total'] >= 60 else RED)
    score_data = [['Metric', 'Score', 'Status']]
    for k, v in score['breakdown'].items():
        status = '✓ Good' if v >= 80 else ('⚠ Fair' if v >= 60 else '✗ Poor')
        score_data.append([k.replace('_', ' ').title(), f"{v:.0f}/100", status])
    score_data.append(['TOTAL SCORE', f"{score['total']:.0f}/100", ''])

    score_table = Table(score_data, colWidths=[6*cm, 4*cm, 6*cm])
    score_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DARK),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#F0FDF4')),
        ('TEXTCOLOR', (1, -1), (1, -1), score_color),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#E2E8F0')),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -2),
         [colors.white, colors.HexColor('#F8FAFC')]),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 0.4*cm))

    # Vitals summary
    story.append(Paragraph("24-Hour Vitals Summary", h2_style))
    avg = df.mean(numeric_only=True)
    std = df.std(numeric_only=True)

    vitals_rows = [['Metric', 'Mean ± SD', 'Min', 'Max', 'Normal Range']]
    vitals_meta = [
        ('heart_rate',   'Heart Rate (bpm)',    '50–100'),
        ('spo2',         'SpO₂ (%)',            '95–100'),
        ('temperature',  'Temperature (°C)',    '36.1–37.5'),
        ('bp_systolic',  'Systolic BP (mmHg)',  '90–130'),
        ('bp_diastolic', 'Diastolic BP (mmHg)', '60–85'),
        ('hrv_ms',       'HRV (ms)',            '20–80'),
    ]
    for col, label, normal in vitals_meta:
        if col in df.columns:
            vitals_rows.append([
                label,
                f"{avg[col]:.1f} ± {std[col]:.1f}",
                f"{df[col].min():.1f}",
                f"{df[col].max():.1f}",
                normal
            ])

    v_table = Table(vitals_rows, colWidths=[4.5*cm, 3.5*cm, 2*cm, 2*cm, 3.5*cm])
    v_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DARK),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#E2E8F0')),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.white, colors.HexColor('#F8FAFC')]),
    ]))
    story.append(v_table)
    story.append(Spacer(1, 0.4*cm))

    # Chart
    if os.path.exists(chart_path):
        story.append(Paragraph("Vitals Trend Chart", h2_style))
        story.append(Image(chart_path, width=16*cm, height=9*cm))
        story.append(Spacer(1, 0.3*cm))

    # Anomaly summary
    if predictions is not None and 'final_anomaly' in predictions.columns:
        story.append(Paragraph("ML Anomaly Detection Results", h2_style))
        n_anom = int(predictions['final_anomaly'].sum())
        total_w = len(predictions)
        rate = predictions['final_anomaly'].mean() * 100

        anom_summary = [
            ['Windows Analyzed', 'Anomalies Detected', 'Anomaly Rate', 'Avg Confidence'],
            [str(total_w), str(n_anom), f"{rate:.1f}%",
             f"{predictions['confidence'].mean()*100:.0f}%"]
        ]
        a_table = Table(anom_summary, colWidths=[4*cm, 4*cm, 3.5*cm, 4*cm])
        a_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), DARK),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('BACKGROUND', (0, 1), (-1, 1),
             RED if rate > 10 else (colors.HexColor('#FEF3C7') if rate > 5 else colors.HexColor('#F0FDF4'))),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#E2E8F0')),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        story.append(a_table)

    # Recommendations
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("Clinical Recommendations", h2_style))
    recs = []
    avg_bp = avg.get('bp_systolic', 120)
    avg_spo2 = avg.get('spo2', 97)
    avg_hrv = avg.get('hrv_ms', 45)

    if avg_bp > 130:
        recs.append("• Elevated blood pressure detected. Consider dietary sodium reduction and follow-up with a physician.")
    if avg_spo2 < 95:
        recs.append("• SpO₂ levels below optimal. Assess for respiratory conditions; supplemental oxygen evaluation recommended.")
    if avg_hrv < 25:
        recs.append("• Low HRV indicates elevated autonomic stress. Stress management techniques and sleep hygiene review advised.")
    if not recs:
        recs.append("• Vitals within acceptable ranges. Continue regular monitoring and maintain current health practices.")

    recs.append("• This report is generated by an automated system. Always consult a qualified healthcare professional.")

    for rec in recs:
        story.append(Paragraph(rec, body_style))

    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width='100%', thickness=1, color=GRAY))
    story.append(Paragraph(
        f"Generated by Health Monitoring System v1.0 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ParagraphStyle('Footer', parent=body_style, fontSize=8,
                       textColor=GRAY, alignment=TA_CENTER)
    ))

    doc.build(story)
    print(f"PDF report saved: {out_path}")
    return out_path


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from data.sensor_simulator import generate_vitals

    df = generate_vitals(n_samples=1440)
    patient = {'id': 'PT-001', 'name': 'Demo Patient',
               'age': 45, 'gender': 'Female'}
    generate_pdf_report(patient, df, out_path='reports/demo_report.pdf')
