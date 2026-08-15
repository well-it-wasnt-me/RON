"""Command-line entry points."""

from robot.cli.doctor import main as doctor
from robot.cli.entrypoint import main as main

__all__ = ["doctor", "main"]
