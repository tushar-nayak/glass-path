from __future__ import annotations


def resolve_device(requested: str | None = None) -> str:
    """Resolve the requested training device.

    Supported values:
    - `auto`: prefer MPS, then CUDA, then CPU
    - `mps`, `cuda`, `cpu`: explicit selection
    """
    requested = (requested or "auto").lower()
    if requested != "auto":
        return requested

    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
