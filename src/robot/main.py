"""CLI entry point for ``python -m robot``."""

from __future__ import annotations

import anyio

from robot.app import DeskBotApp
from robot.config import load_settings
from robot.logging import configure_logging, get_logger

_log = get_logger("main")


def main() -> None:
    """Run the DeskBot application until interrupted."""
    settings = load_settings()
    configure_logging(settings)
    app = DeskBotApp.from_settings(settings)

    async def _run() -> None:
        async with app.run():
            _log.info("main.idle", message="press Ctrl+C to stop")
            await anyio.sleep_forever()

    anyio.run(_run)


if __name__ == "__main__":
    main()
