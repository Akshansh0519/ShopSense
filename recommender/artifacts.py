import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib


MODEL_FILENAMES = {
    "popularity": "popularity.joblib",
    "item_item_raw": "item_item_raw.joblib",
    "item_item_decay": "item_item_decay.joblib",
    "als": "als.joblib",
    "bpr": "bpr.joblib",
    "content": "content.joblib",
    "hybrid": "hybrid.joblib",
    "hybrid_mmr": "hybrid_mmr.joblib",
}


class ArtifactError(RuntimeError):
    """Raised when required production artifacts are missing or invalid."""


def artifact_root(project_root: Path) -> Path:
    return project_root / "artifacts"


def model_dir(project_root: Path) -> Path:
    return artifact_root(project_root) / "models"


def manifest_path(project_root: Path) -> Path:
    return artifact_root(project_root) / "model_manifest.json"


def save_model(project_root: Path, name: str, model: Any) -> Path:
    if name not in MODEL_FILENAMES:
        raise KeyError(f"Unknown model artifact name: {name}")
    target_dir = model_dir(project_root)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / MODEL_FILENAMES[name]
    joblib.dump(model, target)
    return target


def load_model(project_root: Path, name: str) -> Any:
    if name not in MODEL_FILENAMES:
        raise KeyError(f"Unknown model artifact name: {name}")
    target = model_dir(project_root) / MODEL_FILENAMES[name]
    if not target.exists():
        raise ArtifactError(f"Missing model artifact: {target}")
    return joblib.load(target)


def save_manifest(project_root: Path, manifest: Dict[str, Any]) -> Path:
    target = manifest_path(project_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        **manifest,
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def load_manifest(project_root: Path) -> Dict[str, Any]:
    target = manifest_path(project_root)
    if not target.exists():
        raise ArtifactError(f"Missing model manifest: {target}")
    return json.loads(target.read_text(encoding="utf-8"))


def find_project_root(start: Optional[Path] = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "configs" / "default.yaml").exists():
            return candidate
    return current
