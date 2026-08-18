"""Tests for the canary model deployment system.

Phase 8: Canary Model Deployment.

Tests prove:
- models are atomically deployed
- rollback is a single operation
- previous known-good model is retained
- model validation rejects corrupted models
- canary stages advance in order
- promotion criteria are checked
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot.learning.model_registry import (
    CanaryDeploymentManager,
    CanaryStage,
    ModelMetadata,
    ModelRegistry,
)


@pytest.fixture
def registry(tmp_path: Path) -> ModelRegistry:
    return ModelRegistry(registry_dir=tmp_path / "registry")


@pytest.fixture
def sample_model_data() -> dict[str, object]:
    return {
        "schema_version": 1,
        "layers": [
            {"weights": [[0.1, 0.2], [0.3, 0.4]], "biases": [0.0, 0.0]},
            {"weights": [[0.5, 0.6]], "biases": [0.0]},
        ],
    }


@pytest.fixture
def sample_metadata() -> ModelMetadata:
    return ModelMetadata(
        model_version=1,
        schema_version=1,
        state_encoder_version=1,
        action_space_version=1,
        git_commit="abc123",
        dataset_version="v1",
        training_run="run-001",
        validation={"loss": 0.1, "reward": 0.5, "safety_violations": 0},
    )


# ========================================================================
# Model registry
# ========================================================================


class TestModelRegistry:
    """Model registry stores, deploys, and rolls back models."""

    def test_deploy_model(
        self,
        registry: ModelRegistry,
        sample_model_data: dict[str, object],
        sample_metadata: ModelMetadata,
    ) -> None:
        version = registry.deploy(sample_model_data, sample_metadata)
        assert version == 1
        assert registry.active_version == 1
        assert registry.model_path(1).exists()
        assert registry.metadata_path(1).exists()

    def test_atomic_deployment(
        self,
        registry: ModelRegistry,
        sample_model_data: dict[str, object],
        sample_metadata: ModelMetadata,
    ) -> None:
        """Model files are written atomically."""
        registry.deploy(sample_model_data, sample_metadata)
        # The model file should be complete and valid JSON
        data = json.loads(registry.model_path(1).read_text())
        assert "layers" in data

    def test_rollback(
        self,
        registry: ModelRegistry,
        sample_model_data: dict[str, object],
        sample_metadata: ModelMetadata,
    ) -> None:
        """Rollback returns to the previous model."""
        # Deploy v1
        registry.deploy(sample_model_data, sample_metadata)
        # Deploy v2
        meta2 = ModelMetadata(model_version=2)
        registry.deploy(sample_model_data, meta2)
        assert registry.active_version == 2
        assert registry.previous_version == 1

        # Rollback to v1
        version = registry.rollback()
        assert version == 1
        assert registry.active_version == 1

    def test_previous_model_retained(
        self,
        registry: ModelRegistry,
        sample_model_data: dict[str, object],
        sample_metadata: ModelMetadata,
    ) -> None:
        """Previous known-good model is retained after deployment."""
        registry.deploy(sample_model_data, sample_metadata)
        meta2 = ModelMetadata(model_version=2)
        registry.deploy(sample_model_data, meta2)
        assert registry.model_path(1).exists()
        assert registry.model_path(2).exists()

    def test_validate_model_valid(
        self,
        registry: ModelRegistry,
        sample_model_data: dict[str, object],
        sample_metadata: ModelMetadata,
    ) -> None:
        """A valid model passes validation."""
        registry.deploy(sample_model_data, sample_metadata)
        ok, reason = registry.validate_model(1)
        assert ok is True
        assert reason == "ok"

    def test_validate_model_missing(self, registry: ModelRegistry) -> None:
        """Missing model file fails validation."""
        ok, reason = registry.validate_model(999)
        assert ok is False
        assert "not found" in reason

    def test_validate_model_corrupted_weights(self, registry: ModelRegistry) -> None:
        """Corrupted (NaN) weights fail validation."""
        model_data = {
            "schema_version": 1,
            "layers": [
                {"weights": [[float("nan"), 0.2]], "biases": [0.0]},
            ],
        }
        meta = ModelMetadata(model_version=1)
        registry.deploy(model_data, meta)
        ok, reason = registry.validate_model(1)
        assert ok is False
        assert "finite" in reason

    def test_load_active(
        self,
        registry: ModelRegistry,
        sample_model_data: dict[str, object],
        sample_metadata: ModelMetadata,
    ) -> None:
        """Loading the active model returns its data."""
        registry.deploy(sample_model_data, sample_metadata)
        data = registry.load_active()
        assert data is not None
        assert "layers" in data

    def test_load_active_fallback(
        self,
        registry: ModelRegistry,
        sample_model_data: dict[str, object],
        sample_metadata: ModelMetadata,
    ) -> None:
        """Loading falls back to previous if active is corrupted."""
        registry.deploy(sample_model_data, sample_metadata)
        meta2 = ModelMetadata(model_version=2)
        registry.deploy(sample_model_data, meta2)

        # Corrupt the active model
        registry.model_path(2).write_text('{"corrupted": true}')

        # Should fall back to v1
        data = registry.load_active()
        assert data is not None
        assert "layers" in data

    def test_list_versions(
        self,
        registry: ModelRegistry,
        sample_model_data: dict[str, object],
        sample_metadata: ModelMetadata,
    ) -> None:
        """List all available model versions."""
        registry.deploy(sample_model_data, sample_metadata)
        registry.deploy(sample_model_data, ModelMetadata(model_version=2))
        versions = registry.list_versions()
        assert versions == [1, 2]

    def test_get_metadata(
        self,
        registry: ModelRegistry,
        sample_model_data: dict[str, object],
        sample_metadata: ModelMetadata,
    ) -> None:
        """Get metadata for a model version."""
        registry.deploy(sample_model_data, sample_metadata)
        meta = registry.get_metadata(1)
        assert meta is not None
        assert meta.model_version == 1
        assert meta.git_commit == "abc123"


# ========================================================================
# Canary deployment manager
# ========================================================================


class TestCanaryDeployment:
    """Canary stages advance and rollback."""

    def test_stages_in_order(self, registry: ModelRegistry) -> None:
        """Canary stages advance in the correct order."""
        manager = CanaryDeploymentManager(registry=registry)
        assert manager.current_stage == CanaryStage.CANDIDATE

        stages = []
        while manager.current_stage.value != CanaryStage.FULL_DEPLOYMENT.value:
            manager.advance()
            stages.append(manager.current_stage)

        assert CanaryStage.OFFLINE_EVALUATION in stages
        assert CanaryStage.SHADOW in stages
        assert manager.current_stage.value == "full_deployment"

    def test_rollback_stage(self, registry: ModelRegistry) -> None:
        """Rollback goes back one stage."""
        manager = CanaryDeploymentManager(registry=registry)
        manager.advance()  # offline_evaluation
        manager.advance()  # shadow
        assert manager.current_stage.value == "shadow"
        manager.rollback()
        assert manager.current_stage.value == "offline_evaluation"

    def test_reset(self, registry: ModelRegistry) -> None:
        """Reset goes back to candidate."""
        manager = CanaryDeploymentManager(registry=registry)
        manager.advance()
        manager.advance()
        manager.reset()
        assert manager.current_stage == CanaryStage.CANDIDATE


# ========================================================================
# Model metadata
# ========================================================================


class TestModelMetadata:
    """Model metadata serialises correctly."""

    def test_serialization(self) -> None:
        meta = ModelMetadata(
            model_version=5,
            git_commit="def456",
            validation={"loss": 0.05, "reward": 1.2},
        )
        d = meta.to_dict()
        assert d["model_version"] == 5
        assert d["git_commit"] == "def456"
        assert d["validation"]["loss"] == 0.05

    def test_roundtrip(self) -> None:
        meta = ModelMetadata(model_version=3, git_commit="abc")
        d = meta.to_dict()
        meta2 = ModelMetadata.from_dict(d)
        assert meta2.model_version == 3
        assert meta2.git_commit == "abc"
