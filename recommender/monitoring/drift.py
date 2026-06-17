import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def generate_drift_report(train_df: pd.DataFrame, test_df: pd.DataFrame, articles_df: pd.DataFrame, output_path: str = "reports/drift_report.html"):
    logger.info("Generating Evidently drift report...")
    
    # Merge with articles to get categorical features for drift detection
    train_merged = train_df.merge(articles_df, left_on='item_idx', right_on='item_idx', how='left')
    test_merged = test_df.merge(articles_df, left_on='item_idx', right_on='item_idx', how='left')
    
    # Select columns to monitor
    monitor_cols = ['price', 'product_group_name', 'department_name', 'colour_group_name']
    
    # Keep only available columns
    available_cols = [c for c in monitor_cols if c in train_merged.columns]
    
    train_data = train_merged[available_cols].copy()
    test_data = test_merged[available_cols].copy()
    
    # Handle NaNs for Evidently
    train_data = train_data.fillna("Unknown")
    test_data = test_data.fillna("Unknown")
    
    report = Report(metrics=[
        DataDriftPreset(),
    ])
    
    logger.info("Running Evidently analysis...")
    report.run(reference_data=train_data, current_data=test_data)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(output_path)
    logger.info(f"Drift report saved to {output_path}")

if __name__ == "__main__":
    print("Drift report module.")
