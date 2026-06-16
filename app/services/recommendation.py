import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from app.schemas.response import RecommendationItem, RecommendationResponse, SignalDecomposition
from app.services.cache import RedisCache
from recommender.artifacts import ArtifactError, find_project_root, load_manifest, load_model
from recommender.serving.explainability import ReasonGenerator


logger = logging.getLogger(__name__)


class RecommendationService:
    def __init__(self, cache: RedisCache, project_root: Optional[Path] = None):
        self.cache = cache
        self.project_root = find_project_root(project_root)
        self.reason_generator = ReasonGenerator()
        self.model = None
        self.popularity_model = None
        self.manifest = None
        self.user_mapping = {}
        self.reverse_item_mapping = {}
        self.articles = None
        self.model_version = "unloaded"

    def _load_pickle(self, name: str):
        path = self.project_root / "artifacts" / name
        if not path.exists():
            raise ArtifactError(f"Missing required artifact: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def load_artifacts(self):
        if self.model is not None:
            return

        self.manifest = load_manifest(self.project_root)
        active_model = self.manifest.get("active_model", "hybrid_mmr")
        self.model_version = self.manifest.get("model_version", active_model)
        self.model = load_model(self.project_root, active_model)
        self.popularity_model = load_model(self.project_root, "popularity")
        self.user_mapping = self._load_pickle("user_mapping.pkl")
        self.reverse_item_mapping = self._load_pickle("reverse_item_mapping.pkl")

        articles_path = self.project_root / "data" / "raw" / "articles.csv"
        if articles_path.exists():
            self.articles = pd.read_csv(articles_path)
        logger.info("Loaded recommendation artifacts for model version %s", self.model_version)

    def _external_item_id(self, item_idx: int) -> str:
        return str(self.reverse_item_mapping.get(int(item_idx), item_idx))

    def _normalize_signals(self, signals: Dict[str, float]) -> Dict[str, float]:
        best_cf = "als"
        if self.manifest:
            best_cf = self.manifest.get("best_cf_model", "als")
        normalized = {"als": 0.0, "bpr": 0.0, "content": 0.0, "popularity": 0.0, "freshness": 0.0}
        for key, value in (signals or {}).items():
            if key == "cf":
                normalized[best_cf] = float(value)
            elif key in normalized:
                normalized[key] = float(value)
        return normalized

    def _decorate(self, user_id: str, raw_recs: List[Dict], cached: bool, start_time: float) -> RecommendationResponse:
        items = []
        for rank, rec in enumerate(raw_recs, start=1):
            signals = self._normalize_signals(rec.get("signals", {}))
            reason = rec.get("reason") or self.reason_generator.generate_reason(
                signals,
                is_mmr_adjusted=bool(rec.get("mmr_adjusted", False)),
            )
            items.append(
                RecommendationItem(
                    item_id=str(rec["item_id"]),
                    rank=rank,
                    score=float(rec.get("score", 0.0)),
                    reason=reason,
                    signals=SignalDecomposition(**signals),
                )
            )

        return RecommendationResponse(
            user_id=user_id,
            model_version=self.model_version,
            cached=cached,
            recommendations=items,
            latency_ms=(time.time() - start_time) * 1000,
        )

    def _category_popularity(self, category: Optional[str], k: int) -> List[Dict]:
        recs = self.popularity_model.recommend(0, k=max(k * 20, 100), exclude_seen=False)
        if category and self.articles is not None and "product_group_name" in self.articles.columns:
            article_ids = {str(v) for v in self.articles.loc[self.articles["product_group_name"] == category, "article_id"].tolist()}
            recs = [(item_idx, score) for item_idx, score in recs if self._external_item_id(item_idx) in article_ids]

        return [
            {
                "item_id": self._external_item_id(item_idx),
                "score": score,
                "signals": {"popularity": 1.0},
            }
            for item_idx, score in recs[:k]
        ]

    def get_recommendations(self, user_id: str, k: int = 10, category: Optional[str] = None) -> RecommendationResponse:
        start_time = time.time()
        self.load_artifacts()

        cached_recs = self.cache.get_recommendations(user_id, self.model_version)
        if cached_recs:
            return self._decorate(user_id, cached_recs[:k], cached=True, start_time=start_time)

        if user_id not in self.user_mapping:
            raw_recs = self._category_popularity(category, k)
        else:
            user_idx = int(self.user_mapping[user_id])
            if hasattr(self.model, "recommend_with_signals"):
                model_recs = self.model.recommend_with_signals(user_idx, k=k)
                raw_recs = [
                    {
                        "item_id": self._external_item_id(rec["item_idx"]),
                        "score": rec["score"],
                        "signals": rec.get("signals", {}),
                        "mmr_adjusted": rec.get("mmr_adjusted", False),
                    }
                    for rec in model_recs
                ]
            else:
                model_recs = self.model.recommend(user_idx, k=k)
                raw_recs = [
                    {
                        "item_id": self._external_item_id(item_idx),
                        "score": score,
                        "signals": {},
                    }
                    for item_idx, score in model_recs
                ]

        self.cache.set_recommendations(user_id, self.model_version, raw_recs)
        return self._decorate(user_id, raw_recs, cached=False, start_time=start_time)
