from fastapi.testclient import TestClient

from app.main import app
from app.services.recommendation import RecommendationService
from app.api import routes


class NullCache:
    def get_recommendations(self, user_id, model_version):
        return None

    def set_recommendations(self, user_id, model_version, recommendations):
        return None


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_missing_artifacts_returns_503(tmp_path):
    original_service = routes.rec_service
    routes.rec_service = RecommendationService(cache=NullCache(), project_root=tmp_path)
    try:
        client = TestClient(app)
        response = client.get("/recommendations/user_1?k=3")
    finally:
        routes.rec_service = original_service

    assert response.status_code == 503
    assert "Run python scripts/train_all.py first" in response.json()["detail"]
