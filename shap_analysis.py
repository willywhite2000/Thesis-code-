"""
=============================================================================
SHAP ANALYSIS  — Feature Importance for the best Enriched XGBoost model
=============================================================================
Master Thesis - Wilbert | Tilburg University

Loads final_xgb_Enriched.joblib (fitted XGB_SMOTE_weights on full enriched
dataset) and produces:
  1. Global feature importance (mean |SHAP|) — per-class + overall bar plots
  2. Summary (beeswarm) plot for the low-credibility class specifically
  3. CSV with top-30 features ranked by mean |SHAP| per class

Usage:
    python shap_analysis.py
=============================================================================
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
from pathlib import Path
from urllib.parse import urlparse
from collections import Counter
import re
from bs4 import BeautifulSoup

# --- Config (uses the same paths as model_comparison_v4) ---
CONFIG = {
    "html_front_dir": r"C:\Users\Wilbe\OneDrive\Desktop\profiling-data-Copy(1)\html_front",
    "css_front_dir": r"C:\Users\Wilbe\OneDrive\Desktop\profiling-data-Copy(1)\css_front",
    "mbfc_json_path": r"C:\Users\Wilbe\OneDrive\Desktop\profiling-data-Copy(1)\results_expanded.json",
    "enriched_csv": r"C:\Users\Wilbe\OneDrive\Desktop\profiling-data-Copy(1)\thesis_output\low_cred_features_latest.csv",
    "model_results_dir": r"C:\Users\Wilbe\OneDrive\Desktop\profiling-data-Copy(1)\thesis_output\model_results_v4",
    "shap_output_dir": r"C:\Users\Wilbe\OneDrive\Desktop\profiling-data-Copy(1)\thesis_output\model_results_v4\shap",
}

os.makedirs(CONFIG["shap_output_dir"], exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(CONFIG["shap_output_dir"], "shap_analysis.log"), mode="w"),
    ],
)
logger = logging.getLogger(__name__)


# =============================================================================
# REBUILD THE SAME X MATRIX THE MODEL WAS TRAINED ON
# We need the *selected* feature matrix in the same order, BEFORE SMOTE.
# (SHAP is computed on real samples, not synthetic ones.)
# =============================================================================

# Re-import the pipeline's feature machinery by running the same code paths.
# The simplest reliable path: re-load baseline + enriched, recompute features,
# then apply the saved scaler and selector.
#
# We re-use the model_comparison_v4 module to avoid duplicating ~500 lines.
import importlib.util
import sys

# Absolute path to the v4 script in your project folder
V4_PATH = r"C:\Users\Wilbe\OneDrive\Desktop\profiling-data-Copy(1)\model comparison v4"

def _load_v4():
    """Dynamically load the v4 pipeline module so we can reuse its functions."""
    from importlib.machinery import SourceFileLoader
    loader = SourceFileLoader("v4_pipeline", V4_PATH)
    spec = importlib.util.spec_from_loader("v4_pipeline", loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v4_pipeline"] = mod
    loader.exec_module(mod)
    return mod


def rebuild_X_for_shap():
    """
    Rebuild the enriched feature matrix exactly as the pipeline produced it,
    then apply the saved scaler and selector. Returns X_selected (ready for SHAP),
    the class labels y, and the selected feature names.
    """
    logger.info("Loading saved model bundle...")
    bundle = joblib.load(os.path.join(CONFIG["model_results_dir"], "final_xgb_Enriched.joblib"))
    model = bundle["model"]
    scaler = bundle["scaler"]
    selector = bundle["selector"]
    le = bundle["label_encoder"]
    engineered_feats = bundle["feature_names_engineered"]  # 225 features
    selected_feats = bundle["feature_names_selected"]       # 135 features
    class_names = bundle["class_names"]
    logger.info(f"  Model classes: {class_names}")
    logger.info(f"  Engineered features: {len(engineered_feats)}")
    logger.info(f"  Selected features: {len(selected_feats)}")

    # Rebuild the enriched dataset
    logger.info("Rebuilding enriched dataset (this re-extracts HTML features)...")
    v4 = _load_v4()
    entries, mbfc_lookup = v4.load_mbfc_metadata(CONFIG["mbfc_json_path"])
    css_lookup = v4.load_external_css_features(CONFIG["css_front_dir"])
    baseline_df = v4.build_baseline_dataset(CONFIG["html_front_dir"], mbfc_lookup, css_lookup)
    enriched_df = v4.build_enriched_dataset(baseline_df, CONFIG["enriched_csv"], entries)

    # Get shared features (as in main())
    feature_cols = v4.get_feature_columns(baseline_df)
    enriched_feature_cols = v4.get_feature_columns(enriched_df)
    shared_features = sorted(set(feature_cols) & set(enriched_feature_cols))

    # Apply feature engineering (adds 35 derived features -> 225 total)
    engineered_df, full_feature_cols = v4.engineer_features(enriched_df, shared_features)

    # Sanity check: engineered features match what model expects
    if set(full_feature_cols) != set(engineered_feats):
        logger.warning("Engineered feature set differs from saved model's set")
        logger.warning(f"  In current but not saved: {set(full_feature_cols) - set(engineered_feats)}")
        logger.warning(f"  In saved but not current: {set(engineered_feats) - set(full_feature_cols)}")

    # Align columns to match saved ordering
    X_full = engineered_df[engineered_feats].fillna(0).values
    y = le.transform(engineered_df["credibility_class"])

    # Apply scaler then selector (same order as pipeline)
    X_scaled = scaler.transform(X_full)
    X_selected = selector.transform(X_scaled)
    logger.info(f"  X_selected shape: {X_selected.shape}")
    logger.info(f"  Class distribution: {Counter(y)}")

    return X_selected, y, selected_feats, class_names, model


# =============================================================================
# SHAP COMPUTATION & PLOTS
# =============================================================================
def compute_shap_values(model, X, max_samples=1000, random_state=42):
    """
    Compute SHAP values using TreeExplainer (exact for tree ensembles).
    Subsample to max_samples for tractability — 1000 is plenty for stable ranking.
    """
    logger.info(f"\nComputing SHAP values...")
    rng = np.random.RandomState(random_state)
    n = min(max_samples, X.shape[0])
    idx = rng.choice(X.shape[0], size=n, replace=False)
    X_sample = X[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # In newer SHAP versions for multiclass, shap_values is an array of shape
    # (n_samples, n_features, n_classes). In older versions, it's a list of
    # (n_samples, n_features) arrays, one per class. Normalize to the list form.
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # shape (n_samples, n_features, n_classes) -> list over classes
        shap_list = [shap_values[:, :, c] for c in range(shap_values.shape[2])]
    elif isinstance(shap_values, list):
        shap_list = shap_values
    else:
        # binary case
        shap_list = [shap_values]

    logger.info(f"  SHAP values computed for {n} samples, {len(shap_list)} class(es)")
    return shap_list, X_sample, idx


def plot_global_importance(shap_list, feature_names, class_names, output_dir, top_n=20):
    """
    For each class (and overall), plot a horizontal bar chart of the top_n
    features by mean |SHAP|.
    """
    # Overall importance = mean |SHAP| averaged across classes
    overall_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_list], axis=0)
    order = np.argsort(overall_importance)[::-1][:top_n]

    # --- OVERALL plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        [feature_names[i] for i in order[::-1]],
        overall_importance[order[::-1]],
        color="#4477AA",
    )
    ax.set_xlabel("Mean |SHAP value| (averaged across classes)")
    ax.set_title(f"Top {top_n} Features — Overall Importance\n(XGBoost SMOTE+weights, Enriched)")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "shap_global_importance_overall.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved overall plot: {out_path}")

    # --- PER-CLASS plots ---
    colors = {"high credibility": "#228833", "low credibility": "#EE6677", "medium credibility": "#CCBB44"}
    for c_idx, cls in enumerate(class_names):
        imp = np.abs(shap_list[c_idx]).mean(axis=0)
        cls_order = np.argsort(imp)[::-1][:top_n]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(
            [feature_names[i] for i in cls_order[::-1]],
            imp[cls_order[::-1]],
            color=colors.get(cls, "#888888"),
        )
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"Top {top_n} Features — {cls.title()}\n(XGBoost SMOTE+weights, Enriched)")
        plt.tight_layout()
        safe_cls = cls.replace(" ", "_")
        out_path = os.path.join(output_dir, f"shap_global_importance_{safe_cls}.pdf")
        plt.savefig(out_path, bbox_inches="tight")
        plt.savefig(out_path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved {cls} plot: {out_path}")

    return overall_importance, order


def plot_low_cred_beeswarm(shap_list, X_sample, feature_names, class_names, output_dir, top_n=15):
    """
    Beeswarm (summary) plot focused on the low-credibility class.
    Shows per-instance SHAP contributions for the top_n features — this is
    the most interpretable figure for the thesis Results chapter.
    """
    # Find the low-credibility class index
    low_idx = None
    for i, c in enumerate(class_names):
        if "low" in c.lower():
            low_idx = i
            break
    if low_idx is None:
        logger.warning("  No 'low' class found; skipping beeswarm plot")
        return

    sv = shap_list[low_idx]
    imp = np.abs(sv).mean(axis=0)
    top_feats = np.argsort(imp)[::-1][:top_n]

    # Use shap's built-in summary plot, restricted to top features
    fig = plt.figure(figsize=(9, 7))
    shap.summary_plot(
        sv[:, top_feats],
        X_sample[:, top_feats],
        feature_names=[feature_names[i] for i in top_feats],
        show=False,
        plot_size=None,
    )
    plt.title(f"SHAP Summary — Low-Credibility Class (top {top_n} features)")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "shap_beeswarm_low_credibility.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved low-cred beeswarm: {out_path}")


def save_top_features_table(shap_list, feature_names, class_names, output_dir, top_n=30):
    """Save a CSV with top features per class — for reference in the thesis text."""
    rows = []
    for c_idx, cls in enumerate(class_names):
        imp = np.abs(shap_list[c_idx]).mean(axis=0)
        order = np.argsort(imp)[::-1][:top_n]
        for rank, feat_idx in enumerate(order, 1):
            rows.append({
                "class": cls,
                "rank": rank,
                "feature": feature_names[feat_idx],
                "mean_abs_shap": imp[feat_idx],
            })
    # Overall
    overall_imp = np.mean([np.abs(sv).mean(axis=0) for sv in shap_list], axis=0)
    order = np.argsort(overall_imp)[::-1][:top_n]
    for rank, feat_idx in enumerate(order, 1):
        rows.append({
            "class": "OVERALL",
            "rank": rank,
            "feature": feature_names[feat_idx],
            "mean_abs_shap": overall_imp[feat_idx],
        })

    df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, "shap_top_features.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"  Saved top features CSV: {out_path}")
    return df


# =============================================================================
# MAIN
# =============================================================================
def main():
    logger.info("="*70)
    logger.info("SHAP ANALYSIS — Enriched XGBoost (SMOTE + class weights)")
    logger.info("="*70)

    X, y, feature_names, class_names, model = rebuild_X_for_shap()

    shap_list, X_sample, sample_idx = compute_shap_values(model, X, max_samples=1000)

    plot_global_importance(shap_list, feature_names, class_names,
                            CONFIG["shap_output_dir"], top_n=20)
    plot_low_cred_beeswarm(shap_list, X_sample, feature_names, class_names,
                            CONFIG["shap_output_dir"], top_n=15)
    top_df = save_top_features_table(shap_list, feature_names, class_names,
                                      CONFIG["shap_output_dir"], top_n=30)

    # Console preview of top-15 overall + top-15 low-cred
    logger.info("\n" + "="*70)
    logger.info("TOP 15 FEATURES OVERALL:")
    logger.info("="*70)
    overall = top_df[top_df["class"] == "OVERALL"].head(15)
    for _, row in overall.iterrows():
        logger.info(f"  {row['rank']:>2}. {row['feature']:<40s} |SHAP| = {row['mean_abs_shap']:.4f}")

    logger.info("\n" + "="*70)
    logger.info("TOP 15 FEATURES FOR LOW-CREDIBILITY CLASS:")
    logger.info("="*70)
    low = top_df[top_df["class"].str.contains("low", case=False, na=False)].head(15)
    for _, row in low.iterrows():
        logger.info(f"  {row['rank']:>2}. {row['feature']:<40s} |SHAP| = {row['mean_abs_shap']:.4f}")

    logger.info(f"\nAll SHAP outputs saved to: {CONFIG['shap_output_dir']}")
    logger.info("Done.")


if __name__ == "__main__":
    print("""
    +-----------------------------------------------------------+
    |  SHAP ANALYSIS                                            |
    |  Loads final_xgb_Enriched.joblib -> feature importance    |
    |  Master Thesis -- Tilburg University                      |
    +-----------------------------------------------------------+
    """)
    main()