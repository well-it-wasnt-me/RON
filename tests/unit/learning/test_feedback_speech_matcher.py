"""Phase 5: the constrained speech feedback matcher (no LLM).

``ConversationService._match_feedback`` classifies an utterance as positive,
negative, or ``None`` using a small fixed lexicon. It is deliberately
constrained — feedback is never free-form LLM generation.
"""

from __future__ import annotations

import pytest

from robot.services.conversation_service import ConversationService

# ``_match_feedback`` is a static method, so reach it through the class.
match = ConversationService._match_feedback


class TestPositive:
    @pytest.mark.parametrize("text", ["good", "Good!", "yes", "nice", "right", "correct"])
    def test_positive_single_word(self, text: str) -> None:
        assert match(text) == "positive"

    @pytest.mark.parametrize("text", ["that's good", "thats good", "good job"])
    def test_positive_phrases(self, text: str) -> None:
        assert match(text) == "positive"


class TestNegative:
    @pytest.mark.parametrize("text", ["no", "No.", "wrong", "don't", "dont", "nope", "bad", "incorrect"])
    def test_negative_single_word(self, text: str) -> None:
        assert match(text) == "negative"

    @pytest.mark.parametrize("text", ["not that", "no don't", "no dont"])
    def test_negative_phrases(self, text: str) -> None:
        assert match(text) == "negative"


class TestNotFeedback:
    @pytest.mark.parametrize(
        "text",
        [
            "",
            "hello there",
            "what time is it",
            "tell me a joke",
            "wave back",  # teaching-ish, not feedback
            "the weather today",
        ],
    )
    def test_not_feedback(self, text: str) -> None:
        assert match(text) is None

    def test_word_containing_cue_is_not_matched(self) -> None:
        """``goodbye`` starts with ``good`` but is its own token -> not 'good'."""
        # ``goodbye`` is a distinct token, not equal to ``good``.
        assert match("goodbye friend") is None
