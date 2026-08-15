"""Shared fixtures and helpers for integration tests.

Integration tests must not depend on the developer's ``.env`` file,
physical hardware, GPU/CUDA availability, or external services.  These
helpers build deterministic all-mock settings so the tests run anywhere.
"""

from __future__ import annotations

from robot.config import AppSettings


def make_test_settings() -> AppSettings:
    """Build all-mock settings that do not load ``.env``.

    * ``hardware = mock`` → MockMicrophone / MockCamera
    * ``audio = mock`` / ``tts = mock`` → no physical speaker
    * ``vector_memory.enabled = False`` → keyword Memory (no torch/CUDA)
    * ``conversation.store = memory`` → no sqlite connections
    * ``api.enabled = False`` → no background HTTP server
    * ``perception.enabled = False`` → no camera scanning
    """
    settings = AppSettings(_env_file=None, timezone="Europe/Dublin")
    settings.api.enabled = False
    settings.perception.enabled = False
    settings.memory.enabled = True  # keyword Memory - no torch dependency
    settings.vector_memory.enabled = False
    settings.sounds.enabled = False
    return settings
