"""DeskBot interactive TUI - an interactive terminal interface for the robot.

Provides a rich terminal UI (built on Textual) that shows:

* An ASCII-art rendering of the face display (braille characters).
* Live servo positions with gauges.
* A microphone level meter.
* A camera view (when available).
* A scrollable event log.
* A command prompt for interacting with the robot.

Start with::

    deskbot - interactive

Or::

    python -m robot.cli.interactive
"""

from robot.cli.interactive.app import DeskBotTUI

__all__ = ["DeskBotTUI"]
