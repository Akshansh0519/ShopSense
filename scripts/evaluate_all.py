import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from recommender.artifacts import load_model, model_dir
from recommender.evaluation.evaluator import ModelEvaluator
from recommender.evaluation.segments import SegmentEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def reports_exist(reports_dir: Path) -> bool:
    """Check if valid metric reports already exist."""
    metrics_path = reports_dir / "metrics.json"
    segment_path = reports_dir / "segment_metrics.json"

    if not metrics_path.exists() or not segment_path.exists():
        return False

    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        segments = json.loads(segment_path.read_text(encoding="utf-8"))
        # Verify they have actual content, not empty dicts
        has_metrics = bool(metrics.get("metrics"))
        has_segments = bool(segments.get("metrics"))
        return has_metrics and has_segments
    except (json.JSONDecodeError, KeyError):
        return False


def main():
    parser = argparse.ArgumentParser(description="Evaluate all trained ShopSense models.")
    parser.add_argument("--force", action="store_true", help="Re-evaluate even if reports already exist.")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    processed_dir = project_root / "data" / "processed"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ── Check existing reports first ──────────────────────────────────────────
    if not args.force and reports_exist(reports_dir):
        logger.info(
            "Evaluation reports already exist in reports/. "
            "Skipping recalculation. Use --force to re-evaluate."
        )
        # Print a quick summary of what's already there
        metrics = json.loads((reports_dir / "metrics.json").read_text(encoding="utf-8"))
        for model, vals in metrics.get("metrics", {}).items():
            ndcg = vals.get("ndcg@10", 0)
            logger.info("  %s: NDCG@10 = %.4f", model, ndcg)
        return

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading evaluation data...")
    train_df = pd.read_pickle(processed_dir / "train.pkl")
    test_df = pd.read_pickle(processed_dir / "test.pkl")

    # Load models
    models_to_eval = {}
    m_dir = model_dir(project_root)
    for model_name in ["als", "bpr", "popularity", "hybrid_mmr"]:
        try:
            models_to_eval[model_name] = load_model(project_root, model_name)
        except Exception as e:
            logger.warning("Could not load model %s: %s", model_name, e)

    if not models_to_eval:
        logger.error("No models found to evaluate! Run train_all.py first.")
        sys.exit(1)

    num_items = len(pd.read_pickle(project_root / "artifacts" / "item_mapping.pkl"))

    # Limit test users to speed up evaluation for demo
    test_users = test_df["user_idx"].drop_duplicates().head(500)
    test_df_sample = test_df[test_df["user_idx"].isin(test_users)].copy()

    # ── Overall evaluation ────────────────────────────────────────────────────
    evaluator = ModelEvaluator(test_df_sample, num_items=num_items)

    overall_metrics = {}
    logger.info("Starting overall evaluation...")
    for name, model in models_to_eval.items():
        logger.info("Evaluating %s...", name)
        metrics = evaluator.evaluate(model, k=10)
        overall_metrics[name] = metrics

    # ── Segment evaluation ────────────────────────────────────────────────────
    logger.info("Starting segment evaluation...")
    segmenter = SegmentEvaluator()
    segments = segmenter.assign_segments(train_df)

    segment_metrics = {seg: {} for seg in segments.keys()}
    for seg_name, seg_users in segments.items():
        seg_test = segmenter.get_segment_test_data(test_df_sample, seg_users)
        if seg_test.empty:
            continue
        seg_evaluator = ModelEvaluator(seg_test, num_items=num_items)
        for name, model in models_to_eval.items():
            logger.info("Evaluating %s on %s segment...", name, seg_name)
            segment_metrics[seg_name][name] = seg_evaluator.evaluate(model, k=10)

    # ── Write reports ─────────────────────────────────────────────────────────
    with open(reports_dir / "metrics.json", "w") as f:
        json.dump({"metrics": overall_metrics}, f, indent=2)

    with open(reports_dir / "segment_metrics.json", "w") as f:
        json.dump({"metrics": segment_metrics}, f, indent=2)

    logger.info("Evaluation complete. Reports saved to reports/")


if __name__ == "__main__":
    main()
