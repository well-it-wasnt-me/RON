"""Async-safe application lifecycle manager.

The lifecycle is the only place that owns the root task group and the
event-bus subscription. Components subscribe to events in ``startup`` and
unsubscribe in ``shutdown``; the manager guarantees both are called exactly
once and all errors are surfaced.
"""

from __future__ import annotations

# Re-export public names so that ``from robot.lifecycle import Lifecycle``
# continues to work after the module was split into a package.
from robot.lifecycle._core import (
    Lifecycle,
    LifecycleHooks,
    LifecycleState,
    ShutdownHook,
    StartupHook,
)
from robot.lifecycle.degradation import (
    DegradationEntry,
    DegradationRegistry,
    Status,
    safe_init,
)

__all__ = [
    "DegradationEntry",
    "DegradationRegistry",
    "Lifecycle",
    "LifecycleHooks",
    "LifecycleState",
    "ShutdownHook",
    "StartupHook",
    "Status",
    "safe_init",
]
