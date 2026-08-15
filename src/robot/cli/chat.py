"""Interactive text chat interface for DeskBot.

Run with::

    deskbot chat

or::

    python -m robot.cli.chat

Typed messages enter the **same** conversation pipeline used by speech
input — :meth:`ConversationService.handle_user_text` transitions to
LISTENING and publishes :class:`SpeechRecognized`, so the existing
LLM → tool-calling → TTS → state-transition path is exercised
identically.

The interface works with entirely mock backends (no microphone, no
speaker).  The response is always printed to stdout; TTS/audio are
attempted when configured but not required for text output.

Log output is redirected to **stderr** so it never mixes with the chat
text on stdout.  The log level is raised to WARNING so the terminal
stays clean unless something goes wrong.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys

from robot.app import DeskBotApp
from robot.config import load_settings
from robot.events.events import BotReply, LLMTokenReceived
from robot.logging import configure_logging, get_logger

_log = get_logger("cli.chat")

_HELP_TEXT = """\
DeskBot text interface
Type a message and press Enter.
Commands: /quit, /exit, /help, /clear
"""

_QUIT_COMMANDS = frozenset({"/quit", "/exit"})


async def _run_chat(app: DeskBotApp) -> None:  # noqa: PLR0912
    """Run the interactive chat loop inside the running app context."""
    conversation = app.conversation
    if conversation is None:
        print("Error: conversation service is not available.")
        return

    loop = asyncio.get_running_loop()

    # --- response capture --------------------------------------------------
    reply_future: asyncio.Future[BotReply] | None = None
    token_parts: list[str] = []
    streaming_active = False

    def _on_reply(event: BotReply) -> None:
        nonlocal reply_future
        if reply_future is not None and not reply_future.done():
            reply_future.set_result(event)

    def _on_token(event: LLMTokenReceived) -> None:
        nonlocal streaming_active
        if event.token:
            token_parts.append(event.token)
            streaming_active = True
            # Print streamed token immediately (no newline until done).
            print(event.token, end="", flush=True)

    app.bus.subscribe(BotReply, _on_reply)
    app.bus.subscribe(LLMTokenReceived, _on_token)

    print(_HELP_TEXT)

    try:
        while True:
            # Run blocking input() in a thread so the event loop stays
            # responsive for TTS playback, event processing, etc.
            try:
                raw = await loop.run_in_executor(None, lambda: input("> "))
            except (EOFError, KeyboardInterrupt):
                print()
                break

            line = raw.strip()
            if not line:
                continue

            if line in _QUIT_COMMANDS:
                break
            if line == "/help":
                print(_HELP_TEXT)
                continue
            if line == "/clear":
                token_parts.clear()
                streaming_active = False
                print("\033[2J\033[H", end="")  # ANSI clear screen
                print(_HELP_TEXT)
                continue

            # --- conversation turn ----------------------------------------
            token_parts.clear()
            streaming_active = False
            reply_future = loop.create_future()

            try:
                # handle_user_text publishes SpeechRecognized, which
                # triggers _on_speech → LLM → TTS → audio → IDLE.
                # The publish() call awaits the full pipeline, so this
                # blocks until the turn completes.
                await conversation.handle_user_text(line, source="text")
            except Exception as exc:
                # If tokens were streamed, print them as the response.
                if streaming_active:
                    print()  # newline after streamed tokens
                print(f"DeskBot: [error: {exc}]")
                # Ensure state returns to a sane state.
                from robot.behavior.state_machine import RobotState

                with contextlib.suppress(Exception):
                    if conversation.state_machine.state is not RobotState.IDLE:
                        await conversation.state_machine.transition(RobotState.IDLE)
                continue

            # Wait for the BotReply event (should already be resolved
            # because handle_user_text awaits the full pipeline).
            try:
                reply = await asyncio.wait_for(reply_future, timeout=30.0)
            except TimeoutError:
                # If tokens were streamed, display them as the response.
                if streaming_active:
                    print()  # newline after streamed tokens
                    print("[response timed out — showing streamed text]")
                else:
                    print("DeskBot: [no response received]")
                continue

            # If streaming tokens were displayed, they are already on
            # stdout; just print a newline.  Otherwise print the full
            # response.
            if streaming_active:
                print()  # newline after streamed tokens
            else:
                print(f"DeskBot: {reply.text}")

            # Report TTS/audio status if degraded.
            tts_name = type(conversation.tts).__name__
            if tts_name == "MockTTS":
                print("[TTS: mock backend — no physical speech]")
    finally:
        app.bus.unsubscribe(BotReply, _on_reply)
        app.bus.unsubscribe(LLMTokenReceived, _on_token)


def main() -> None:
    """Entry point for ``deskbot chat``."""
    import anyio

    settings = load_settings()

    # Redirect logs to stderr so they don't mix with chat output on
    # stdout.  Raise the level to WARNING for a clean terminal.
    settings.log_level = "WARNING"
    configure_logging(settings, stream=sys.stderr)

    _log.info(
        "chat.start",
        env=settings.env,
        hardware=settings.hardware,
        tts_provider=settings.tts.provider,
    )

    # The chat interface doesn't need the REST API server, the
    # perception loop, or sound effects.  Disabling them avoids
    # port conflicts and unnecessary background work.
    settings.api.enabled = False
    settings.perception.enabled = False
    settings.sounds.enabled = False

    app = DeskBotApp.from_settings(settings)

    async def _run() -> None:
        async with app.run():
            await _run_chat(app)

    with contextlib.suppress(KeyboardInterrupt):
        anyio.run(_run)


if __name__ == "__main__":
    main()
