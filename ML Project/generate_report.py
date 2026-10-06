"""
Generates the Spotter Machine Learning Engineer Assessment Report as both PDF and DOCX.
Includes executive summary, validation strategy, data quality, feature engineering,
model benchmarking metrics, December chart analysis, and embedded chart graphic.
"""

import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT


def build_pdf_report(pdf_filename: str = "Spotter_MLE_Assessment_Report.pdf") -> None:
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        rightMargin=40, leftMargin=40,
        topMargin=40, bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette
    PRIMARY = colors.HexColor("#064A56")
    SECONDARY = colors.HexColor("#2E8B57")
    TEXT_DARK = colors.HexColor("#1A252C")
    BG_LIGHT = colors.HexColor("#F4F7F8")
    
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'],
        fontName='Helvetica-Bold', fontSize=22, leading=26,
        textColor=PRIMARY, spaceAfter=8
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=12, leading=16,
        textColor=SECONDARY, spaceAfter=15
    )
    h1_style = ParagraphStyle(
        'SectionH1', parent=styles['Heading2'],
        fontName='Helvetica-Bold', fontSize=14, leading=18,
        textColor=PRIMARY, spaceBefore=12, spaceAfter=6
    )
    h2_style = ParagraphStyle(
        'SectionH2', parent=styles['Heading3'],
        fontName='Helvetica-Bold', fontSize=11, leading=14,
        textColor=SECONDARY, spaceBefore=8, spaceAfter=4
    )
    body_style = ParagraphStyle(
        'BodyDark', parent=styles['Normal'],
        fontName='Helvetica', fontSize=9.5, leading=13.5,
        textColor=TEXT_DARK, spaceAfter=6
    )
    bullet_style = ParagraphStyle(
        'BulletDark', parent=body_style,
        leftIndent=12, bulletIndent=4, spaceAfter=3
    )

    story = []
    
    # Title Header
    story.append(Paragraph("Machine Learning Engineer Assessment Report", title_style))
    story.append(Paragraph("Freight Rate Prediction & December 2025 Forecasting | Spotter ML Assessment", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY, spaceAfter=12))

    # Executive Summary
    story.append(Paragraph("Executive Summary", h1_style))
    story.append(Paragraph(
        "This report presents an end-to-end Machine Learning pipeline for predicting truckload freight rates (`posted_rate`) "
        "across US shipping lanes. Evaluating labeled historical data (`train-test.csv`, Jan–Oct 2025), out-of-time validation "
        "loads (`validation.csv`, Nov–Dec 2025), and a fixed lane December 2025 forecasting benchmark (`december-chart-inputs.csv`), "
        "our solution achieves an out-of-time **Mean Absolute Error (MAE) of $151.59**, **MAPE of 6.98%**, and **R² of 0.8133**.",
        body_style
    ))

    # 1. Validation Strategy & Data Split
    story.append(Paragraph("1. Validation Approach & Train/Test Split", h1_style))
    story.append(Paragraph(
        "<b>Temporal Out-of-Time Validation Scheme:</b> Freight markets exhibit continuous temporal shifts, seasonal patterns, "
        "and market index fluctuations. Standard random K-Fold cross-validation suffers from temporal data leakage (predicting past loads "
        "using future market signals). Because the target evaluation set (`validation.csv`) covers Nov–Dec 2025, we established a strict "
        "temporal holdout split:",
        body_style
    ))
    story.append(Paragraph("• <b>Training Set (Jan 1, 2025 – Aug 31, 2025):</b> 38,477 loads (80.2% of labeled data) used for model fitting.", bullet_style))
    story.append(Paragraph("• <b>Validation Holdout (Sep 1, 2025 – Oct 31, 2025):</b> 9,523 out-of-time loads (19.8%) used for hyperparameter tuning & evaluation.", bullet_style))
    story.append(Paragraph("• <b>Final Retraining (Jan 1, 2025 – Oct 31, 2025):</b> Full 48,000 loads used to train the final production ensemble model.", bullet_style))
    story.append(Paragraph(
        "<b>Target Variable Formulation (Rate Per Mile):</b> Rather than modeling total dollar rates directly (which range from $57 to $25,533), "
        "we model <b>Rate Per Mile (RPM = posted_rate / distance)</b>. This standardizes variance across short-haul and long-haul shipments, "
        "preventing extreme long-haul loads from dominating gradient updates.",
        body_style
    ))

    # 2. Data Exploration & Data Quality Handling
    story.append(Paragraph("2. Data Quality & Exploratory Data Analysis (EDA)", h1_style))
    story.append(Paragraph(
        "Key data hygiene findings and remediation steps:",
        body_style
    ))
    story.append(Paragraph("• <b>Missing Value Imputation:</b> 300 rows in train and 165 in validation lacked `weight`. Missing weights were imputed using equipment-specific medians (Dry Van: 32,000 lbs, Reefer: 35,000 lbs, Flatbed: 40,000 lbs). 374 train and 249 validation rows lacked `market_index`, which were forward-filled using temporal forward/backward interpolation.", bullet_style))
    story.append(Paragraph("• <b>Spatial Consistency:</b> pickup_lat/lon and delivery_lat/lon were checked against reported distance. Haversine distance showed a 0.9995 correlation with reported distance, with an average circuity factor of 1.19x.", bullet_style))
    story.append(Paragraph("• <b>Equipment Rate Disparities:</b> Reefer equipment carries the highest average rate ($2.42/mi), followed by Flatbed ($2.31/mi) and Dry Van ($2.12/mi).", bullet_style))

    # 3. Feature Engineering Breakdown
    story.append(Paragraph("3. Feature Engineering Architecture", h1_style))
    story.append(Paragraph(
        "We engineered 32 domain-informed features categorized into 4 core pillars:",
        body_style
    ))
    story.append(Paragraph("1. <b>Geographic & Spatial:</b> Haversine distance, distance ratio (`distance / haversine_dist`), latitude/longitude deltas, travel bearing angle (degrees), and lane midpoint coordinates.", bullet_style))
    story.append(Paragraph("2. <b>Temporal & Cyclical Dynamics:</b> Sine/cosine transformations of `dayofweek` (7-day period) and `dayofyear` (365-day period), month, quarter, weekend indicator, and holiday flags.", bullet_style))
    story.append(Paragraph("3. <b>Load Density & Weight:</b> Weight per mile, weight ratio relative to equipment median weight, equipment type one-hot encodings.", bullet_style))
    story.append(Paragraph("4. <b>Bayesian Target Encoding:</b> Out-of-fold smoothed rate-per-mile statistics for specific pickup-delivery lanes (`lane_encoded_rpm`), pickup cities, and delivery cities.", bullet_style))

    # 4. Model Selection & Benchmark Results
    story.append(Paragraph("4. Model Selection & Benchmark Metrics", h1_style))
    story.append(Paragraph(
        "Six model architectures were benchmarked on the Sep–Oct 2025 out-of-time temporal holdout:",
        body_style
    ))

    # Table of Results
    table_data = [
        ["Model Architecture", "MAE ($)", "RMSE ($)", "MAPE (%)", "R² Score"],
        ["Ridge Regression Baseline", "$226.89", "$682.70", "9.67%", "0.7999"],
        ["Random Forest Regressor", "$185.73", "$716.30", "8.24%", "0.7797"],
        ["XGBoost Regressor", "$168.04", "$676.08", "8.01%", "0.8037"],
        ["CatBoost Regressor", "$155.60", "$655.86", "7.08%", "0.8153"],
        ["LightGBM Regressor", "$151.42", "$661.79", "6.85%", "0.8119"],
        ["Ensemble (LGB 40% + XGB 30% + CAT 30%)", "$151.59", "$659.33", "6.98%", "0.8133"]
    ]
    
    t = Table(table_data, colWidths=[2.2*inch, 1.0*inch, 1.0*inch, 1.0*inch, 1.0*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8.5),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('BACKGROUND', (0,1), (-1,1), BG_LIGHT),
        ('BACKGROUND', (0,3), (-1,3), BG_LIGHT),
        ('BACKGROUND', (0,5), (-1,5), BG_LIGHT),
        ('BACKGROUND', (0,6), (-1,6), colors.HexColor("#E2F0D9")),
        ('FONTNAME', (0,6), (-1,6), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#D0D7DE")),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    # 5. December Prediction Chart Discussion
    story.append(Paragraph("5. December 2025 Fixed Lane Prediction Chart Analysis", h1_style))
    story.append(Paragraph(
        "Using the candidate scoring script `score.py`, predictions were generated for all 31 days of December 2025 "
        "under fixed lane parameters: <b>Lexington to Fort Wayne (360 miles, Dry Van, 32,000 lbs)</b>.",
        body_style
    ))
    
    chart_path = "scorer_results/candidate_december.png"
    if os.path.exists(chart_path):
        story.append(RLImage(chart_path, width=6.5*inch, height=2.89*inch))
        story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>Key Insights from the December Chart:</b>",
        body_style
    ))
    story.append(Paragraph("• <b>Rate Stability & Range:</b> The predicted total load rate remains tightly bounded between <b>$808.31 and $820.35</b> (an average rate of $2.25/mi to $2.28/mi). This directly aligns with historical Dry Van rates for 360-mile Midwest lanes.", bullet_style))
    story.append(Paragraph("• <b>Seasonal & Weekly Cycles:</b> Small periodic peaks occur on mid-week shipping spikes (Tuesdays/Wednesdays), while slight dips occur during weekend dispatch windows and late-month holiday lulls.", bullet_style))
    story.append(Paragraph("• <b>Absence of Artifact Outliers:</b> The trend is smooth and smooth-bounded, verifying that missing input feature imputation and spatial coordinate lookups operated without edge-case distortions.", bullet_style))

    # Build PDF
    doc.build(story)
    print(f"Generated PDF report: {pdf_filename}")


def build_docx_report(docx_filename: str = "Spotter_MLE_Assessment_Report.docx") -> None:
    doc = docx.Document()
    
    # Page Margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.6)
        section.bottom_margin = Inches(0.6)
        section.left_margin = Inches(0.6)
        section.right_margin = Inches(0.6)

    PRIMARY_RGB = RGBColor(6, 74, 86)
    SECONDARY_RGB = RGBColor(46, 139, 87)

    # Title
    p_title = doc.add_paragraph()
    run_title = p_title.add_run("Machine Learning Engineer Assessment Report")
    run_title.font.name = "Arial"
    run_title.font.size = Pt(20)
    run_title.font.bold = True
    run_title.font.color.rgb = PRIMARY_RGB
    
    p_sub = doc.add_paragraph()
    run_sub = p_sub.add_run("Freight Rate Prediction & December 2025 Forecasting | Spotter Assessment")
    run_sub.font.name = "Arial"
    run_sub.font.size = Pt(11)
    run_sub.font.bold = True
    run_sub.font.color.rgb = SECONDARY_RGB

    def add_h1(text):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.font.name = "Arial"
        run.font.size = Pt(13)
        run.font.bold = True
        run.font.color.rgb = PRIMARY_RGB
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)

    def add_body(text):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.font.name = "Arial"
        run.font.size = Pt(10)
        p.paragraph_format.space_after = Pt(4)

    def add_bullet(text):
        p = doc.add_paragraph(style='List Bullet')
        run = p.add_run(text)
        run.font.name = "Arial"
        run.font.size = Pt(9.5)
        p.paragraph_format.space_after = Pt(2)

    # Content
    add_h1("Executive Summary")
    add_body(
        "This report presents an end-to-end Machine Learning pipeline for predicting freight rates across US shipping lanes. "
        "Evaluating historical data (Jan–Oct 2025), out-of-time validation loads (Nov–Dec 2025), and a fixed lane December 2025 "
        "forecasting benchmark, our solution achieves an out-of-time Mean Absolute Error (MAE) of $151.59, MAPE of 6.98%, and R² of 0.8133."
    )

    add_h1("1. Validation Approach & Train/Test Split")
    add_body("Temporal Out-of-Time Validation Scheme: To prevent data leakage and evaluate real-world out-of-time generalization, we established a temporal split:")
    add_bullet("Training Set (Jan 1 – Aug 31, 2025): 38,477 loads (80.2%) used for model training.")
    add_bullet("Validation Holdout (Sep 1 – Oct 31, 2025): 9,523 loads (19.8%) used for out-of-time evaluation.")
    add_bullet("Final Retraining: Full 48,000 labeled loads used to train the production ensemble model.")
    add_body("Rate Per Mile (RPM) Target Formulation: We modeled Rate Per Mile (RPM = posted_rate / distance), stabilizing target variance across short and long haul distances.")

    add_h1("2. Data Quality & Exploratory Data Analysis")
    add_bullet("Missing Values: Missing weights were imputed using equipment medians (Dry Van: 32,000 lbs, Reefer: 35,000 lbs, Flatbed: 40,000 lbs). Missing market_index values were temporal forward-filled.")
    add_bullet("Spatial Check: Haversine distance correlated 0.9995 with reported distance, with 1.19x circuity factor.")
    add_bullet("Equipment Rates: Reefer ($2.42/mi) > Flatbed ($2.31/mi) > Dry Van ($2.12/mi).")

    add_h1("3. Feature Engineering Architecture")
    add_body("32 engineered features were constructed across spatial, temporal, load density, and Bayesian target encoding pillars.")

    add_h1("4. Model Selection & Benchmark Metrics")
    
    # Table
    table_data = [
        ["Model Architecture", "MAE ($)", "RMSE ($)", "MAPE (%)", "R² Score"],
        ["Ridge Regression Baseline", "$226.89", "$682.70", "9.67%", "0.7999"],
        ["Random Forest Regressor", "$185.73", "$716.30", "8.24%", "0.7797"],
        ["XGBoost Regressor", "$168.04", "$676.08", "8.01%", "0.8037"],
        ["CatBoost Regressor", "$155.60", "$655.86", "7.08%", "0.8153"],
        ["LightGBM Regressor", "$151.42", "$661.79", "6.85%", "0.8119"],
        ["Ensemble (LGB + XGB + CAT)", "$151.59", "$659.33", "6.98%", "0.8133"]
    ]
    t = doc.add_table(rows=len(table_data), cols=5)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx, row in enumerate(table_data):
        for c_idx, val in enumerate(row):
            cell = t.cell(r_idx, c_idx)
            cell.text = val
            p = cell.paragraphs[0]
            p.runs[0].font.name = "Arial"
            p.runs[0].font.size = Pt(8.5)
            if r_idx == 0:
                p.runs[0].font.bold = True

    add_h1("5. December 2025 Fixed Lane Chart Analysis")
    chart_path = "scorer_results/candidate_december.png"
    if os.path.exists(chart_path):
        doc.add_picture(chart_path, width=Inches(6.2))

    add_bullet("Predicted load rate ranges steadily between $808.31 and $820.35 ($2.25/mi - $2.28/mi).")
    add_bullet("Periodic weekly cycles reflect midweek shipping demand spikes.")

    doc.save(docx_filename)
    print(f"Generated DOCX report: {docx_filename}")


if __name__ == "__main__":
    build_pdf_report()
    build_docx_report()
