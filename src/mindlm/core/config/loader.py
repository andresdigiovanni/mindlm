from pathlib import Path

import yaml

from mindlm.core.config.models import RAGConfig


def load_config(path: Path) -> RAGConfig:
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} is empty or not a valid YAML mapping")
    return RAGConfig(**data)
