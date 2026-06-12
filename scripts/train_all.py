import logging
import sys
import argparse
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from recommender.artifacts import ArtifactError, load_model, save_manifest, save_model
from recommender.data.interaction_matrix import InteractionMatrixBuilder
from recommender.data.loader import DataLoader
from recommender.data.preprocessor import DataPreprocessor
from recommender.data.splitter import DataSplitter
from recommender.evaluation.evaluator import ModelEvaluator
from recommender.experiments.tracker import ExperimentTracker
from recommender.models.als_model import ALSRecommender
from recommender.models.bpr_model import BPRRecommender
from recommender.models.content import ContentRecommender
from recommender.models.hybrid import HybridRecommender
from recommender.models.item_item import ItemItemRecommender
from recommender.models.mmr import MMRRecommender
from recommender.models.popularity import PopularityRecommender


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_and_log(tracker: ExperimentTracker, run_name: str, model, matrix, params: dict):
    with tracker.start_run(run_name):
        tracker.log_params(params)
        model.fit(matrix)
    return model


def limit_users(df, max_users):
    if not max_users or max_users <= 0:
        return df
    users = df["user_idx"].drop_duplicates().head(max_users)
    return df[df["user_idx"].isin(users)].copy()


def model_exists(project_root: Path, name: str) -> bool:
    from recommender.artifacts import MODEL_FILENAMES, model_dir

    return (model_dir(project_root) / MODEL_FILENAMES[name]).exists()


def main():
    parser = argparse.ArgumentParser(description="Train ShopSense models and persist artifacts.")
    parser.add_argument("--full", action="store_true", help="Use full validation when choosing ALS vs BPR.")
    parser.add_argument("--force", action="store_true", help="Retrain models even when artifacts already exist.")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    config_path = project_root / "configs" / "default.yaml"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    (project_root / "artifacts" / "models").mkdir(parents=True, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("Starting training pipeline...")

    preprocessor = DataPreprocessor(artifacts_dir=str(project_root / "artifacts"))
    splits_exist = all((processed_dir / name).exists() for name in ["train.pkl", "val.pkl", "test.pkl"])
    mappings_exist = all((project_root / "artifacts" / name).exists() for name in ["user_mapping.pkl", "item_mapping.pkl"])

    if splits_exist and mappings_exist and not args.force:
        logger.info("Reusing existing processed splits and mappings.")
        train_df = np.nan
        val_df = np.nan
        test_df = np.nan
        import pandas as pd

        train_df = pd.read_pickle(processed_dir / "train.pkl")
        val_df = pd.read_pickle(processed_dir / "val.pkl")
        test_df = pd.read_pickle(processed_dir / "test.pkl")
        preprocessor.load_mappings()
        loader = DataLoader(data_dir=str(project_root / "data" / "raw"))
        articles = loader.load_articles()
    else:
        loader = DataLoader(data_dir=str(project_root / "data" / "raw"))
        transactions = loader.load_transactions()
        articles = loader.load_articles()

        filtered_tx = preprocessor.filter_interactions(
            transactions,
            min_user_interactions=config["data"]["min_user_interactions"],
            min_item_interactions=config["data"]["min_item_interactions"],
        )
        mapped_tx = preprocessor.create_mappings(filtered_tx)
        preprocessor.save_mappings()

        splitter = DataSplitter(
            train_ratio=config["data"]["split_ratios"]["train"],
            val_ratio=config["data"]["split_ratios"]["val"],
            test_ratio=config["data"]["split_ratios"]["test"],
        )
        train_df, val_df, test_df = splitter.split(mapped_tx)
        train_df.to_pickle(processed_dir / "train.pkl")
        val_df.to_pickle(processed_dir / "val.pkl")
        test_df.to_pickle(processed_dir / "test.pkl")

    num_users = len(preprocessor.user_mapping)
    num_items = len(preprocessor.item_mapping)
    needs_raw_matrix = args.force or any(
        not model_exists(project_root, name)
        for name in ["popularity", "item_item_raw", "bpr", "content"]
    )
    needs_decay_matrix = args.force or any(
        not model_exists(project_root, name)
        for name in ["item_item_decay", "als"]
    )
    raw_train_matrix = None
    decay_train_matrix = None
    matrix_builder = None
    if needs_raw_matrix or needs_decay_matrix:
        matrix_builder = InteractionMatrixBuilder(num_users, num_items)
    if needs_raw_matrix:
        raw_train_matrix = matrix_builder.build_raw_matrix(train_df)
    if needs_decay_matrix:
        decay_train_matrix = matrix_builder.build_time_decay_matrix(
            train_df,
            decay_rate=config["data"]["time_decay_rate"],
        )

    tracker = ExperimentTracker()

    if model_exists(project_root, "popularity") and not args.force:
        popularity = load_model(project_root, "popularity")
    else:
        popularity = PopularityRecommender()
        train_and_log(tracker, "baseline_popularity", popularity, raw_train_matrix, {"model": "popularity"})
        save_model(project_root, "popularity", popularity)

    if model_exists(project_root, "item_item_raw") and not args.force:
        item_item_raw = load_model(project_root, "item_item_raw")
    else:
        item_item_raw = ItemItemRecommender()
        train_and_log(tracker, "item_item_raw", item_item_raw, raw_train_matrix, {"model": "item_item", "matrix": "raw"})
        save_model(project_root, "item_item_raw", item_item_raw)

    if model_exists(project_root, "item_item_decay") and not args.force:
        item_item_decay = load_model(project_root, "item_item_decay")
    else:
        item_item_decay = ItemItemRecommender()
        train_and_log(tracker, "item_item_decay", item_item_decay, decay_train_matrix, {"model": "item_item", "matrix": "time_decay"})
        save_model(project_root, "item_item_decay", item_item_decay)

    if model_exists(project_root, "als") and not args.force:
        als = load_model(project_root, "als")
    else:
        als = ALSRecommender(
            factors=config["models"]["als"]["factors"],
            regularization=config["models"]["als"]["regularization"],
            iterations=config["models"]["als"]["iterations"],
            alpha=config["models"]["als"]["alpha"],
        )
        train_and_log(
            tracker,
            "als_implicit",
            als,
            decay_train_matrix,
            {
                "model": "als",
                "factors": config["models"]["als"]["factors"],
                "regularization": config["models"]["als"]["regularization"],
                "iterations": config["models"]["als"]["iterations"],
                "alpha": config["models"]["als"]["alpha"],
            },
        )
        save_model(project_root, "als", als)

    if model_exists(project_root, "bpr") and not args.force:
        bpr = load_model(project_root, "bpr")
    else:
        bpr = BPRRecommender(
            factors=config["models"]["bpr"]["factors"],
            regularization=config["models"]["bpr"]["regularization"],
            iterations=config["models"]["bpr"]["iterations"],
        )
        train_and_log(
            tracker,
            "bpr_implicit",
            bpr,
            raw_train_matrix,
            {
                "model": "bpr",
                "factors": config["models"]["bpr"]["factors"],
                "regularization": config["models"]["bpr"]["regularization"],
                "iterations": config["models"]["bpr"]["iterations"],
            },
        )
        save_model(project_root, "bpr", bpr)

    if model_exists(project_root, "content") and not args.force:
        content = load_model(project_root, "content")
    else:
        content = ContentRecommender(model_name=config["models"]["content"]["embedding_model"])
        with tracker.start_run("content_embeddings"):
            tracker.log_params({"model": "content", "embedding_model": config["models"]["content"]["embedding_model"]})
            content.fit(
                raw_train_matrix,
                articles_df=articles,
                item_mapping=preprocessor.item_mapping,
                artifacts_dir=str(project_root / "artifacts"),
            )
        if content.item_embeddings is not None:
            np.save(project_root / "artifacts" / "item_embeddings.npy", content.item_embeddings)
        save_model(project_root, "content", content)

    validation_limit = None if args.full else config.get("evaluation", {}).get("max_validation_users", 1000)
    val_eval_df = limit_users(val_df, validation_limit)
    logger.info("Selecting best CF model on %s validation users.", val_eval_df["user_idx"].nunique())
    val_evaluator = ModelEvaluator(val_eval_df, num_items=num_items)
    als_val = val_evaluator.evaluate(als, k=10)
    bpr_val = val_evaluator.evaluate(bpr, k=10)
    best_cf_name = "als" if als_val.get("ndcg@10", 0.0) >= bpr_val.get("ndcg@10", 0.0) else "bpr"
    best_cf_model = als if best_cf_name == "als" else bpr
    logger.info("Best CF model on validation NDCG@10: %s", best_cf_name)

    weights = config["models"]["hybrid"]["weights"]
    hybrid = HybridRecommender(
        models={
            "cf": best_cf_model,
            "content": content,
            "popularity": popularity,
        },
        weights={
            "cf": weights["cf"],
            "content": weights["content"],
            "popularity": weights["popularity"],
        },
    )
    save_model(project_root, "hybrid", hybrid)

    hybrid_mmr = MMRRecommender(
        base_model=hybrid,
        item_embeddings=content.item_embeddings,
        lambda_param=config["models"]["mmr"]["lambda_param"],
        candidate_pool_size=config["models"]["mmr"]["candidate_pool_size"],
    )
    save_model(project_root, "hybrid_mmr", hybrid_mmr)

    save_manifest(
        project_root,
        {
            "active_model": "hybrid_mmr",
            "model_version": "hybrid_mmr_v1",
            "best_cf_model": best_cf_name,
            "dataset": "h_and_m_personalized_fashion",
            "num_users": num_users,
            "num_items": num_items,
            "artifacts": {
                "models_dir": "artifacts/models",
                "item_embeddings": "artifacts/item_embeddings.npy",
                "train_split": "data/processed/train.pkl",
                "val_split": "data/processed/val.pkl",
                "test_split": "data/processed/test.pkl",
            },
            "validation": {
                "als": als_val,
                "bpr": bpr_val,
            },
        },
    )

    logger.info("Training complete. Artifacts written under artifacts/ and artifacts/models/.")


if __name__ == "__main__":
    main()
