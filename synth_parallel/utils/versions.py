from __future__ import annotations

import importlib
import importlib.metadata
from typing import Dict, Optional


def _get_version(package: str) -> Optional[str]:
    try:
        return importlib.metadata.version(package)
    except Exception:
        return None


def collect_versions() -> Dict[str, Optional[str]]:
    packages = [
        "torch",
        "transformers",
        "datasets",
        "openai",
        "vllm",
        "synth-parallel",
    ]
    versions = {pkg: _get_version(pkg) for pkg in packages}
    return versions
