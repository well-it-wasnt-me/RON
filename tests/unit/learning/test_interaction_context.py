"""Phase 4: InteractionContext holds and tags teaching interaction identifiers.

The context only holds the active identifiers — it never auto-mints in response
to actions. Each ``begin_*`` mints a fresh uuid4; the matching ``end_*`` clears
it. ``current_metadata`` omits ``None`` values so ambient (non-teaching)
actions produce an empty dict and the executor's merge is a no-op.

Attribute reads are funnelled through locals before asserting, so the field's
declared ``str | None`` type stays visible to mypy across the mutating
``begin_*``/``end_*`` calls (mypy does not reset attribute narrowing through
method calls, which would otherwise mark the post-mutation checks unreachable).
"""

from __future__ import annotations

import uuid

from robot.learning.interaction_context import InteractionContext


class TestMintingAndClearing:
    def test_begin_interaction_mints_uuid(self) -> None:
        ctx = InteractionContext()
        before = ctx.interaction_id
        assert before is None
        new_id = ctx.begin_interaction()
        uuid.UUID(new_id)  # a real uuid4 string, parseable
        after = ctx.interaction_id
        assert after == new_id
        ctx.end_interaction()
        ended = ctx.interaction_id
        assert ended is None

    def test_begin_teaching_session_mints_uuid(self) -> None:
        ctx = InteractionContext()
        before = ctx.teaching_session_id
        assert before is None
        new_id = ctx.begin_teaching_session()
        uuid.UUID(new_id)
        after = ctx.teaching_session_id
        assert after == new_id
        ctx.end_teaching_session()
        ended = ctx.teaching_session_id
        assert ended is None

    def test_begin_episode_mints_uuid(self) -> None:
        ctx = InteractionContext()
        before = ctx.episode_id
        assert before is None
        new_id = ctx.begin_episode()
        uuid.UUID(new_id)
        after = ctx.episode_id
        assert after == new_id
        ctx.end_episode()
        ended = ctx.episode_id
        assert ended is None

    def test_mints_are_distinct(self) -> None:
        ctx = InteractionContext()
        a = ctx.begin_interaction()
        ctx.end_interaction()
        b = ctx.begin_interaction()
        assert a != b

    def test_reset_clears_all(self) -> None:
        ctx = InteractionContext()
        ctx.begin_interaction()
        ctx.begin_teaching_session()
        ctx.begin_episode()
        i = ctx.interaction_id
        s = ctx.teaching_session_id
        e = ctx.episode_id
        assert i is not None
        assert s is not None
        assert e is not None
        ctx.reset()
        ri = ctx.interaction_id
        rs = ctx.teaching_session_id
        re = ctx.episode_id
        assert ri is None
        assert rs is None
        assert re is None


class TestCurrentMetadata:
    def test_empty_when_nothing_active(self) -> None:
        ctx = InteractionContext()
        # Ambient (non-teaching) state -> empty dict -> executor merge is a no-op.
        assert ctx.current_metadata() == {}

    def test_omits_none_keys(self) -> None:
        ctx = InteractionContext()
        interaction = ctx.begin_interaction()
        # Only the active interaction id is present.
        assert ctx.current_metadata() == {"interaction_id": interaction}

    def test_full_metadata_when_all_active(self) -> None:
        ctx = InteractionContext()
        interaction = ctx.begin_interaction()
        session = ctx.begin_teaching_session()
        episode = ctx.begin_episode()
        assert ctx.current_metadata() == {
            "interaction_id": interaction,
            "teaching_session_id": session,
            "episode_id": episode,
        }

    def test_partial_after_end(self) -> None:
        ctx = InteractionContext()
        ctx.begin_interaction()
        session = ctx.begin_teaching_session()
        ctx.end_interaction()
        # interaction cleared -> only the teaching session remains.
        assert ctx.current_metadata() == {"teaching_session_id": session}

    def test_does_not_auto_mint_on_read(self) -> None:
        """Reading metadata never mints — ambient actions stay untagged."""
        ctx = InteractionContext()
        for _ in range(5):
            assert ctx.current_metadata() == {}
        i = ctx.interaction_id
        s = ctx.teaching_session_id
        e = ctx.episode_id
        assert i is None
        assert s is None
        assert e is None
