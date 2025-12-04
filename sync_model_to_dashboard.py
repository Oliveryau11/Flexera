"""
Sync MLModel outputs to Dashboard public/api folder
Run this after training the model to update the dashboard with latest predictions and feedback data.

Usage:
    python sync_model_to_dashboard.py
"""

import shutil
from pathlib import Path

# Paths
MLMODEL_DIR = Path(__file__).parent / "MLModel"
DASHBOARD_API_DIR = Path(__file__).parent / "dashboard" / "public" / "api"

# Files to sync from MLModel root
ROOT_FILES = [
    "model_metrics.csv",
    "win_probabilities.csv",
    "model_registry.json",
    "RealDummyData.xlsx",  # Also sync the data file if updated
]

# Feedback files to sync (these go into api/feedback/)
FEEDBACK_FILES = [
    "calibration_table.csv",
    "feature_importance.csv",
    "feature_importance_grouped.csv",
    "segment_perf_regions.csv",
    "segment_perf_dealtypes.csv",
    "segment_perf_region_x_saletype.csv",
    "loss_reasons_overall.csv",
    "loss_reasons_by_region.csv",
    "loss_reasons_by_country.csv",
    "loss_drivers.csv",
    "used_features.csv",
    "error_analysis_val.csv",
]


def sync():
    """Copy MLModel outputs to dashboard public/api directory."""
    
    # Ensure target directories exist
    DASHBOARD_API_DIR.mkdir(parents=True, exist_ok=True)
    feedback_target = DASHBOARD_API_DIR / "feedback"
    feedback_target.mkdir(exist_ok=True)
    
    print(f"Syncing MLModel outputs to {DASHBOARD_API_DIR}")
    print("-" * 60)
    
    # Sync root files
    for fname in ROOT_FILES:
        src = MLMODEL_DIR / fname
        dst = DASHBOARD_API_DIR / fname
        if src.exists():
            shutil.copy2(src, dst)
            print(f"[OK] {fname}")
        else:
            print(f"[SKIP] {fname} (not found)")
    
    # Sync feedback files
    print("\nFeedback files:")
    feedback_src = MLMODEL_DIR / "feedback"
    for fname in FEEDBACK_FILES:
        src = feedback_src / fname
        dst = feedback_target / fname
        if src.exists():
            shutil.copy2(src, dst)
            print(f"[OK] feedback/{fname}")
        else:
            print(f"[SKIP] feedback/{fname} (not found)")
    
    print("-" * 60)
    print("Sync complete! Dashboard will now use the latest model outputs.")
    print("\nTo start the dashboard:")
    print("  cd dashboard && npm run dev")


if __name__ == "__main__":
    sync()





