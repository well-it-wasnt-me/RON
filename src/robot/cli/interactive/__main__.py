"""Entry point for ``python -m robot.cli.interactive``."""

from __future__ import annotations

from robot.cli.interactive.app import run_interactive


def main() -> None:
    """Run the interactive TUI."""
    run_interactive()


if __name__ == "__main__":
    main()
