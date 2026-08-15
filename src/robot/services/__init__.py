"""Higher-level application services.

Services compose several components to deliver a feature (e.g. a
``ConversationService`` glues together STT, LLM, TTS). They live in
``services/`` so that they can be tested independently from the rest of the
app.
"""

from robot.services.conversation_service import ConversationService
from robot.services.executor import ActionExecutor

__all__ = ["ActionExecutor", "ConversationService"]
