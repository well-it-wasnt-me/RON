"""``deskbot`` console script entry point.

Supports subcommands:

* ``deskbot``       — run the robot application (default)
* ``deskbot chat``  — interactive text chat interface

If no subcommand is recognised, the full application starts as before.
"""

from __future__ import annotations

import sys

from robot.config import load_settings
from robot.logging import configure_logging, get_logger

_log = get_logger("cli.entrypoint")


def main() -> None:
    """Run the DeskBot application or a subcommand."""
    args = sys.argv[1:]

    if args and args[0] == "chat":
        from robot.cli.chat import main as chat_main

        chat_main()
        return

    # Default: run the full application.
    settings = load_settings()
    configure_logging(settings)
    _log.info(
        "cli.start",
        env=settings.env,
        log_level=settings.log_level,
        hardware=settings.hardware,
    )
    from robot.main import main as _main

    _main()


if __name__ == "__main__":
    main()
