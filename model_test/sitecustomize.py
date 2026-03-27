"""Project-local Python startup customizations for model_test.

This module is imported automatically by Python when ``model_test`` is on
``PYTHONPATH``. We use it to steer ONNX Runtime session creation toward GPU
execution by default while preserving explicit per-call overrides.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, List, Optional


def _split_providers(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _cpu_only(providers: Iterable[Any]) -> bool:
    names = [str(item) for item in providers]
    return bool(names) and all(name == "CPUExecutionProvider" for name in names)


def _preferred_providers(ort: Any) -> List[str]:
    raw = os.environ.get("MODEL_TEST_ORT_PROVIDERS", "").strip()
    if raw:
        preferred = _split_providers(raw)
    else:
        preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    available = set(ort.get_available_providers())
    resolved = [provider for provider in preferred if provider in available]

    if not resolved and "CPUExecutionProvider" in available:
        resolved = ["CPUExecutionProvider"]
    return resolved


def _patch_onnxruntime() -> None:
    if os.environ.get("MODEL_TEST_ORT_FORCE_CPU", "0") == "1":
        return

    try:
        import onnxruntime as ort  # type: ignore
    except Exception:
        return

    if getattr(ort, "_npux_inference_session_patched", False):
        return

    original = ort.InferenceSession

    def patched_inference_session(*args: Any, **kwargs: Any) -> Any:
        providers: Optional[Any] = kwargs.get("providers")
        provider_options = kwargs.get("provider_options")

        if providers is None and len(args) >= 3:
            providers = args[2]

        should_override = providers is None or (
            isinstance(providers, (list, tuple)) and _cpu_only(providers)
        )

        if should_override:
            resolved = _preferred_providers(ort)
            if resolved:
                kwargs["providers"] = resolved
                if provider_options is None and len(resolved) > 1:
                    kwargs.setdefault("provider_options", None)

        return original(*args, **kwargs)

    ort.InferenceSession = patched_inference_session  # type: ignore[attr-defined]
    ort._npux_inference_session_patched = True  # type: ignore[attr-defined]


_patch_onnxruntime()
