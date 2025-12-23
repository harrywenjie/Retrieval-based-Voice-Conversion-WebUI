from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

try:  # Pydantic v2 only
    from pydantic import ConfigDict  # type: ignore
except Exception:  # pragma: no cover - v1 fallback
    ConfigDict = None  # type: ignore

_COMPAT_DIR = Path(__file__).resolve().parent


def apply_pre_import_patches() -> None:
    """Patches that must run before importing gradio.*"""
    _ensure_gradio_client_serializing()


def apply_post_import_patches() -> None:
    """Patches that should run after importing gradio.*"""
    _ensure_predict_body_defaults()


def _ensure_gradio_client_serializing() -> None:
    """Provide the legacy gradio_client.serializing module if new versions removed it."""
    try:
        import gradio_client.serializing  # noqa: F401
        return
    except Exception:
        pass

    compat_module = _COMPAT_DIR / "gradio_client_serializing.py"
    if not compat_module.exists():
        return

    spec = importlib.util.spec_from_file_location(
        "gradio_client.serializing", compat_module
    )
    if spec is None or spec.loader is None:
        return

    module = importlib.util.module_from_spec(spec)
    sys.modules["gradio_client.serializing"] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]


def _ensure_predict_body_defaults() -> None:
    """Patch gradio.data_classes.PredictBody to stay compatible with Pydantic v2."""
    if not _is_pydantic_v2():
        return

    try:
        import gradio.data_classes as data_classes
        import gradio.queueing as queueing
    except Exception:
        return

    class PredictBody(BaseModel):  # type: ignore[misc]
        if ConfigDict is not None:  # pragma: no branch - only runs on v2+
            model_config = ConfigDict(extra="allow")  # type: ignore[attr-defined]
        else:  # pragma: no cover - v1 fallback
            class Config:
                extra = "allow"

        session_hash: Optional[str] = None
        event_id: Optional[str] = None
        data: List[Any] = Field(default_factory=list)
        event_data: Optional[Any] = None
        fn_index: Optional[int] = None
        batched: bool = False
        request: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None

    data_classes.PredictBody = PredictBody
    queueing.PredictBody = PredictBody


def _is_pydantic_v2() -> bool:
    try:
        import pydantic
    except Exception:
        return False

    version = getattr(pydantic, "__version__", "0")
    major = _extract_major(version)
    return major >= 2


def _extract_major(version: str) -> int:
    token = version.split(".")[0]
    digits = "".join(ch for ch in token if ch.isdigit())
    return int(digits) if digits else 0


__all__ = [
    "apply_pre_import_patches",
    "apply_post_import_patches",
]
