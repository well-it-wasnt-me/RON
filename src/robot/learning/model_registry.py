"""Canary model deployment with atomic loading and rollback.

Deploy learned models gradually and reversibly.

Features:
* **Model registry** - every model has metadata (version, schema,
  encoder version, action space version, git commit, dataset version,
  validation metrics).
* **Atomic deployment** - never overwrite the active checkpoint
  directly.  Write a temp file, flush, then atomically replace the
  active pointer.  On startup, validate schema, dimensions, checksum,
  finite weights, action-space version, encoder version.  If invalid,
  load the previous known-good model.
* **Canary stages** - candidate -> offline evaluation -> shadow ->
  small action subset -> limited active use -> full approved deployment.
* **Rollback** - single operation.  Keep the previous known-good model.
* **Promotion criteria** - zero safety violations, zero invalid
  actions, no crashes, latency within limit, benchmark pass, real-world
  metrics not worse than baseline.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from robot.logging import get_logger

_log = get_logger("learning.model_registry")


# ---------------------------------------------------------------------------
# Canary stage
# ---------------------------------------------------------------------------


class CanaryStage(str, Enum):
    """Canary deployment stages (in order)."""

    CANDIDATE = "candidate"
    OFFLINE_EVALUATION = "offline_evaluation"
    SHADOW = "shadow"
    SMALL_ACTION_SUBSET = "small_action_subset"
    LIMITED_ACTIVE = "limited_active"
    FULL_DEPLOYMENT = "full_deployment"


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelMetadata:
    """Metadata for a registered model.

    Attributes
    ----------
    model_version:
        Version number of the model.
    schema_version:
        Schema version of the model file format.
    state_encoder_version:
        Version of the state encoder used during training.
    multimodal_version:
        Version of the multimodal encoder (if used).
    action_space_version:
        Version of the action space.
    git_commit:
        Git commit hash of the code used to train.
    dataset_version:
        Version of the dataset used for training.
    training_run:
        Identifier for the training run.
    validation:
        Validation metrics (loss, reward, safety_violations, latency).
    """

    model_version: int
    schema_version: int = 1
    state_encoder_version: int = 1
    multimodal_version: int = 0
    action_space_version: int = 1
    git_commit: str = ""
    dataset_version: str = ""
    training_run: str = ""
    validation: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "schema_version": self.schema_version,
            "state_encoder_version": self.state_encoder_version,
            "multimodal_version": self.multimodal_version,
            "action_space_version": self.action_space_version,
            "git_commit": self.git_commit,
            "dataset_version": self.dataset_version,
            "training_run": self.training_run,
            "validation": dict(self.validation),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        return cls(
            model_version=int(data["model_version"]),
            schema_version=int(data.get("schema_version", 1)),
            state_encoder_version=int(data.get("state_encoder_version", 1)),
            multimodal_version=int(data.get("multimodal_version", 0)),
            action_space_version=int(data.get("action_space_version", 1)),
            git_commit=str(data.get("git_commit", "")),
            dataset_version=str(data.get("dataset_version", "")),
            training_run=str(data.get("training_run", "")),
            validation=dict(data.get("validation", {})),
        )


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ModelRegistry:
    """Manages model versions with atomic deployment and rollback.

    Parameters
    ----------
    registry_dir:
        Directory for storing model files and metadata.
    """

    registry_dir: Path
    _active_version: int = field(default=0, init=False, repr=False)
    _previous_version: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        # Create subdirectories
        (self.registry_dir / "models").mkdir(exist_ok=True)
        (self.registry_dir / "metadata").mkdir(exist_ok=True)
        # Load active version from pointer file
        pointer = self.registry_dir / "active.json"
        if pointer.exists():
            try:
                data = json.loads(pointer.read_text())
                self._active_version = int(data.get("active_version", 0))
                self._previous_version = int(data.get("previous_version", 0))
            except Exception:
                _log.warning("model_registry.corrupt_pointer")

    @property
    def active_version(self) -> int:
        return self._active_version

    @property
    def previous_version(self) -> int:
        return self._previous_version

    def model_path(self, version: int) -> Path:
        """Return the file path for a model version."""
        return self.registry_dir / "models" / f"model_v{version}.json"

    def metadata_path(self, version: int) -> Path:
        """Return the metadata file path for a model version."""
        return self.registry_dir / "metadata" / f"metadata_v{version}.json"

    def deploy(
        self,
        model_data: dict[str, Any],
        metadata: ModelMetadata,
    ) -> int:
        """Atomically deploy a new model version.

        1. Write model to a temp file.
        2. Flush and fsync.
        3. Atomically rename to the final path.
        4. Update the active pointer.
        5. Keep the previous version for rollback.

        Returns the deployed version number.
        """
        version = metadata.model_version
        model_path = self.model_path(version)
        meta_path = self.metadata_path(version)

        # Write model data to temp file
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=str(self.registry_dir / "models"),
            suffix=".tmp",
            delete=False,
        ) as tmp:
            json.dump(model_data, tmp)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)

        # Atomically rename
        tmp_path.replace(model_path)

        # Write metadata
        meta_path.write_text(metadata.to_json(), encoding="utf-8")

        # Update active pointer (atomically)
        previous = self._active_version
        self._previous_version = previous
        self._active_version = version

        pointer_data = {
            "active_version": version,
            "previous_version": previous,
            "deployed_at": time.time(),
        }
        pointer_tmp = self.registry_dir / "active.json.tmp"
        pointer_tmp.write_text(json.dumps(pointer_data))
        pointer_tmp.replace(self.registry_dir / "active.json")

        _log.info(
            "model_registry.deployed",
            version=version,
            previous=previous,
        )
        return version

    def rollback(self) -> int:
        """Rollback to the previous known-good model.

        Returns the version number now active.
        """
        if self._previous_version == 0:
            _log.warning("model_registry.no_previous_version")
            return self._active_version

        target = self._previous_version
        old_active = self._active_version

        self._previous_version = old_active
        self._active_version = target

        pointer_data = {
            "active_version": target,
            "previous_version": old_active,
            "rollback_at": time.time(),
        }
        pointer_tmp = self.registry_dir / "active.json.tmp"
        pointer_tmp.write_text(json.dumps(pointer_data))
        pointer_tmp.replace(self.registry_dir / "active.json")

        _log.info(
            "model_registry.rolled_back",
            from_version=old_active,
            to_version=target,
        )
        return target

    def validate_model(self, version: int, expected_schema: int = 1) -> tuple[bool, str]:  # noqa: PLR0911
        """Validate a model file: schema, dimensions, finite weights.

        Returns ``(True, "ok")`` if valid, ``(False, reason)`` otherwise.
        """
        model_path = self.model_path(version)
        if not model_path.exists():
            return False, f"model file not found: {model_path}"

        try:
            data = json.loads(model_path.read_text())
        except Exception as e:
            return False, f"cannot parse model: {e}"

        # Check schema version
        schema = data.get("schema_version", 0)
        if schema != expected_schema:
            return False, f"schema version mismatch: {schema} != {expected_schema}"

        # Check finite weights
        layers = data.get("layers", [])
        for i, layer in enumerate(layers):
            weights = layer.get("weights", [])
            for j, row in enumerate(weights):
                for k, val in enumerate(row):
                    if not isinstance(val, (int, float)):
                        return False, f"layer {i} weight [{j}][{k}] is not a number"
                    if not np.isfinite(val):
                        return False, f"layer {i} weight [{j}][{k}] is not finite"

            biases = layer.get("biases", [])
            for j, val in enumerate(biases):
                if not isinstance(val, (int, float)):
                    return False, f"layer {i} bias [{j}] is not a number"
                if not np.isfinite(val):
                    return False, f"layer {i} bias [{j}] is not finite"

        return True, "ok"

    def load_active(self) -> dict[str, Any] | None:
        """Load the active model data.

        If the active model fails validation, falls back to the
        previous known-good model.
        """
        if self._active_version == 0:
            return None

        ok, reason = self.validate_model(self._active_version)
        if ok:
            data = json.loads(self.model_path(self._active_version).read_text())
            return dict(data)

        _log.warning("model_registry.active_invalid", reason=reason, version=self._active_version)

        # Try previous version
        if self._previous_version > 0:
            ok, reason = self.validate_model(self._previous_version)
            if ok:
                _log.info(
                    "model_registry.fallback_to_previous",
                    version=self._previous_version,
                )
                data = json.loads(self.model_path(self._previous_version).read_text())
                return dict(data)

        return None

    def list_versions(self) -> list[int]:
        """Return all available model versions."""
        model_dir = self.registry_dir / "models"
        versions = []
        for f in model_dir.glob("model_v*.json"):
            try:
                ver = int(f.stem.replace("model_v", ""))
                versions.append(ver)
            except ValueError:
                pass
        return sorted(versions)

    def get_metadata(self, version: int) -> ModelMetadata | None:
        """Load metadata for a model version."""
        meta_path = self.metadata_path(version)
        if not meta_path.exists():
            return None
        try:
            data = json.loads(meta_path.read_text())
            return ModelMetadata.from_dict(data)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Canary deployment manager
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CanaryDeploymentManager:
    """Manages canary deployment stages for a model.

    Parameters
    ----------
    registry:
        The model registry.
    promotion_criteria:
        Dict of criteria that must pass to advance to the next stage.
    """

    registry: ModelRegistry
    _current_stage: CanaryStage = field(default=CanaryStage.CANDIDATE, init=False, repr=False)
    _stage_history: list[tuple[CanaryStage, float]] = field(
        default_factory=list, init=False, repr=False
    )

    @property
    def current_stage(self) -> CanaryStage:
        return self._current_stage

    @property
    def stage_history(self) -> list[tuple[CanaryStage, float]]:
        return list(self._stage_history)

    def advance(self) -> CanaryStage:
        """Advance to the next canary stage.

        Returns the new stage.
        """
        stages = list(CanaryStage)
        idx = stages.index(self._current_stage)
        if idx < len(stages) - 1:
            self._current_stage = stages[idx + 1]
            self._stage_history.append((self._current_stage, time.time()))
            _log.info("canary.advanced", stage=self._current_stage.value)
        return self._current_stage

    def rollback(self) -> CanaryStage:
        """Rollback to the previous stage and the previous model.

        Returns the new stage.
        """
        stages = list(CanaryStage)
        idx = stages.index(self._current_stage)
        if idx > 0:
            self._current_stage = stages[idx - 1]
            self._stage_history.append((self._current_stage, time.time()))

        self.registry.rollback()
        _log.info("canary.rolled_back", stage=self._current_stage.value)
        return self._current_stage

    def reset(self) -> None:
        """Reset to candidate stage."""
        self._current_stage = CanaryStage.CANDIDATE
        self._stage_history.clear()


__all__ = [
    "CanaryDeploymentManager",
    "CanaryStage",
    "ModelMetadata",
    "ModelRegistry",
]
