import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        config = json.load(f)
    _validate_opencode_model(config)
    return config


def _validate_opencode_model(config: Dict[str, Any]) -> None:
    opencode_cfg = config.get("opencode", {})
    model_ref = str(opencode_cfg.get("model") or "").strip()
    project_dir = str(opencode_cfg.get("project_dir") or "").strip()
    if not model_ref or "/" not in model_ref or not project_dir:
        return

    opencode_json = Path(project_dir) / "opencode.json"
    if not opencode_json.exists():
        return

    try:
        with opencode_json.open("r", encoding="utf-8") as f:
            opencode_project_cfg = json.load(f)
    except Exception:
        return

    provider_name, model_name = model_ref.split("/", 1)
    providers = opencode_project_cfg.get("provider", {})
    provider = providers.get(provider_name) if isinstance(providers, dict) else None
    if not isinstance(provider, dict):
        known = ", ".join(sorted(providers.keys())) if isinstance(providers, dict) else ""
        raise ValueError(
            f"opencode.model references provider `{provider_name}`, but it is not registered in {opencode_json}. "
            f"Known providers: {known}"
        )

    models = provider.get("models", {})
    if isinstance(models, dict) and model_name not in models:
        known = ", ".join(sorted(models.keys()))
        raise ValueError(
            f"opencode.model references `{model_ref}`, but model `{model_name}` is not registered under provider "
            f"`{provider_name}` in {opencode_json}. Known models: {known}"
        )
