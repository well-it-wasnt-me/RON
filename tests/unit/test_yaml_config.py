"""Tests for the YAML configuration loader."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from robot.config import AppSettings, load_settings


@pytest.fixture
def tmp_yaml(tmp_path: Path) -> Path:
    """Write a sample config.yaml and return its path."""
    path = tmp_path / "config.yaml"
    path.write_text(
        """
displays:
  backend: circuitpython
  dc_pin: 25
  spi_hz: 32000000
face:
  theme: vector
llm:
  provider: mock
  model: my-test-model
stt:
  provider: mock
"""
    )
    return path


@pytest.fixture
def moved_env(tmp_path: Path) -> None:  # type: ignore[misc]
    """Move the .env file aside so the dotenv source doesn't override values."""
    env_path = Path(".env")
    backup = tmp_path / "env_backup"
    if env_path.exists():
        shutil.move(str(env_path), str(backup))
    yield
    if backup.exists():
        shutil.move(str(backup), str(env_path))


def test_yaml_loader_is_registered(tmp_yaml: Path, moved_env: object) -> None:
    os.environ["DESKBOT_CONFIG_FILE"] = str(tmp_yaml)
    try:
        s = AppSettings()
        assert s.displays.backend == "circuitpython"
        assert s.displays.dc_pin == 25
        assert s.displays.spi_hz == 32_000_000
        assert s.face.theme == "vector"
        assert s.llm.model == "my-test-model"
    finally:
        del os.environ["DESKBOT_CONFIG_FILE"]


def test_env_overrides_yaml(tmp_yaml: Path) -> None:
    """Environment variables take precedence over YAML."""
    os.environ["DESKBOT_CONFIG_FILE"] = str(tmp_yaml)
    os.environ["DESKBOT_LLM__MODEL"] = "env-wins"
    try:
        s = AppSettings()
        # YAML set model=my-test-model; env sets model=env-wins -> env wins.
        assert s.llm.model == "env-wins"
        # (this assert works only when env var takes precedence)
        # face.theme was not overridden in env -> YAML value used.
        assert s.face.theme == "vector"
    finally:
        del os.environ["DESKBOT_CONFIG_FILE"]
        del os.environ["DESKBOT_LLM__MODEL"]


def test_missing_yaml_file_is_silent(tmp_path: Path) -> None:
    """A missing YAML file should not crash."""
    os.environ["DESKBOT_CONFIG_FILE"] = str(tmp_path / "missing.yaml")
    try:
        s = AppSettings()
        # Defaults from the model apply.
        assert s.displays.backend == "mock"
        assert s.face.theme == "vector"
    finally:
        del os.environ["DESKBOT_CONFIG_FILE"]


def test_load_settings_helper_uses_yaml(tmp_yaml: Path, moved_env: object) -> None:
    os.environ["DESKBOT_CONFIG_FILE"] = str(tmp_yaml)
    try:
        s = load_settings()
        assert s.displays.backend == "circuitpython"
    finally:
        del os.environ["DESKBOT_CONFIG_FILE"]


def test_unspecified_yaml_field_uses_default(tmp_yaml: Path) -> None:
    """YAML files may specify only some fields; missing ones keep defaults."""
    os.environ["DESKBOT_CONFIG_FILE"] = str(tmp_yaml)
    try:
        s = AppSettings()
        # personality.curiosity not in YAML -> default 0.7
        assert s.personality.curiosity == 0.7
    finally:
        del os.environ["DESKBOT_CONFIG_FILE"]
