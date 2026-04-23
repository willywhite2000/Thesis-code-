# Addressing the Low-Credibility Detection Gap

Master's thesis code — Wilbert Kooijman, Tilburg University, 2026.
Supervisor: Dr. Chris Emmery.

## Overview
This repository contains the pipeline for HTML-based news source
credibility classification, including web archive enrichment,
feature engineering, model comparison, and SHAP analysis.

## Scripts
- `enrich_low_credibility.py` — recovers low-credibility source HTML
  from the Internet Archive Wayback Machine.
- `model_comparison_v4.py` — full modeling pipeline: feature extraction,
  Optuna hyperparameter tuning, 5-fold CV, McNemar's test.
- `shap_analysis.py` — SHAP feature importance on the final XGBoost model.

## Running
Requires Python 3.13+ and libraries in `requirements.txt`:
    pip install -r requirements.txt

Run order:
    1. python enrich_low_credibility.py
    2. python model_comparison_v4.py
    3. python shap_analysis.py

## Data
The MBFC dataset used in this thesis was obtained from Dr. Chris
Emmery. Requests for access should be directed to him. All scraped
HTML content was retrieved from publicly accessible websites in
accordance with robots.txt.

## Results
See `results/` for aggregated output tables and the
`thesis_output/model_results_v4/` folder (not committed) for
per-configuration confusion matrices and prediction files.
