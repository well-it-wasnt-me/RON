"""CLI commands for the learning system.

Provides ``deskbot-learning-status``, ``deskbot-learning-train``,
``deskbot-learning-evaluate``, ``deskbot-learning-reset``, and
``deskbot-learning-export`` commands.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_status() -> dict[str, object]:
    """Load learning status from the checkpoint directory."""
    from robot.config import load_settings
    from robot.learning.learning_service import CheckpointManager

    settings = load_settings()
    mgr = CheckpointManager(
        checkpoint_dir=settings.learning.checkpoint_dir,
        keep_last_n=settings.learning.keep_last_n_checkpoints,
    )
    latest = mgr.load_latest(tag="current")
    return {
        "checkpoint_dir": str(mgr._dir),
        "latest_checkpoint": str(latest) if latest else None,
        "version": mgr.version,
    }


def cmd_status(args: argparse.Namespace) -> None:
    """Show learning service status."""
    status = _load_status()
    print("DeskBot Learning Status")
    print("=" * 40)
    for key, value in status.items():
        print(f"  {key}: {value}")


def cmd_train(args: argparse.Namespace) -> None:
    """Force a training cycle."""
    print("Forcing training cycle...")
    print("Note: This requires a running DeskBot instance.")
    print("Use the REST API endpoint POST /api/v1/learning/train instead.")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate the current model."""
    from robot.learning.state_encoder import STATE_SIZE
    from robot.learning.world_model import WorldModel

    model_path = args.model
    if model_path is None:
        print("Error: --model path required for offline evaluation.")
        sys.exit(1)

    model = WorldModel(state_size=STATE_SIZE)
    model.load(model_path)
    print(f"Model loaded from: {model_path}")
    print(f"  State size: {model.state_size}")
    print(f"  Action size: {model.action_size}")
    print(f"  Parameters: {model.param_count()}")


def cmd_reset(args: argparse.Namespace) -> None:
    """Reset learning checkpoints."""
    from robot.config import load_settings

    settings = load_settings()
    checkpoint_dir = Path(settings.learning.checkpoint_dir).expanduser()
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.json"))
        if args.confirm:
            for cp in checkpoints:
                cp.unlink()
            print(f"Deleted {len(checkpoints)} checkpoint(s) from {checkpoint_dir}")
        else:
            print(f"Found {len(checkpoints)} checkpoint(s) in {checkpoint_dir}")
            print("Use --confirm to delete them.")
    else:
        print(f"Checkpoint directory {checkpoint_dir} does not exist.")


def cmd_export(args: argparse.Namespace) -> None:
    """Export learning data."""
    output = args.output or "learning_export.json"
    status = _load_status()
    with Path(output).open("w") as f:
        json.dump({"status": status}, f, indent=2)
    print(f"Exported learning data to {output}")


def main() -> None:
    """Entry point for learning CLI commands."""
    parser = argparse.ArgumentParser(
        prog="deskbot-learning",
        description="DeskBot learning system commands",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # status
    subparsers.add_parser("status", help="Show learning service status")

    # train
    subparsers.add_parser("train", help="Force a training cycle")

    # evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate current model")
    eval_parser.add_argument("--model", type=str, help="Path to model checkpoint")

    # reset
    reset_parser = subparsers.add_parser("reset", help="Reset learning checkpoints")
    reset_parser.add_argument("--confirm", action="store_true", help="Confirm deletion")

    # export
    export_parser = subparsers.add_parser("export", help="Export learning data")
    export_parser.add_argument("--output", type=str, help="Output file path")

    args = parser.parse_args()
    if args.command == "status":
        cmd_status(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "reset":
        cmd_reset(args)
    elif args.command == "export":
        cmd_export(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
