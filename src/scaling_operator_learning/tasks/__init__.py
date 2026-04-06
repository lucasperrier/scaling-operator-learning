"""Task registry and base interface for PDE operator-learning datasets.

Every task module exposes:
    generate_dataset(n_samples, resolution, seed) -> dict
        Returns {'inputs': Tensor, 'outputs': Tensor, 'grid': Tensor,
                 'resolution': int}
"""
from __future__ import annotations

from typing import Any, Callable

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_TASK_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {}


def register_task(name: str):
    """Decorator to register a task's generate_dataset function."""
    def decorator(func: Callable[..., dict[str, Any]]):
        _TASK_REGISTRY[name] = func
        return func
    return decorator


def get_task(name: str) -> Callable[..., dict[str, Any]]:
    if name not in _TASK_REGISTRY:
        raise ValueError(
            f"Unknown task '{name}'. Available: {list(_TASK_REGISTRY.keys())}"
        )
    return _TASK_REGISTRY[name]


def available_tasks() -> list[str]:
    return list(_TASK_REGISTRY.keys())


# Import task modules so they self-register
from . import burgers  # noqa: E402, F401
from . import darcy  # noqa: E402, F401
from . import diffusion  # noqa: E402, F401
from . import diffusion_filtered  # noqa: E402, F401
