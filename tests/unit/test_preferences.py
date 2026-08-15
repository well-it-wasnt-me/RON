"""Tests for user preference learning."""

from robot.ai.preferences import (
    InMemoryPreferenceStore,
    Preference,
    PreferenceTracker,
    SqlitePreferenceStore,
)


class TestPreference:
    def test_create_preference(self) -> None:
        p = Preference(key="humour", value="humorous", confidence=0.8, source="explicit")
        assert p.key == "humour"
        assert p.value == "humorous"
        assert p.confidence == 0.8
        assert p.source == "explicit"


class TestInMemoryPreferenceStore:
    def test_save_and_load(self) -> None:
        store = InMemoryPreferenceStore()
        p = Preference(key="volume", value="high", confidence=0.7)
        store.save(p)
        loaded = store.load("volume")
        assert loaded is not None
        assert loaded.value == "high"

    def test_load_missing(self) -> None:
        store = InMemoryPreferenceStore()
        assert store.load("missing") is None

    def test_load_all(self) -> None:
        store = InMemoryPreferenceStore()
        store.save(Preference(key="a", value="1"))
        store.save(Preference(key="b", value="2"))
        assert len(store.load_all()) == 2

    def test_delete(self) -> None:
        store = InMemoryPreferenceStore()
        store.save(Preference(key="a", value="1"))
        assert store.delete("a") is True
        assert store.load("a") is None
        assert store.delete("a") is False

    def test_upsert(self) -> None:
        store = InMemoryPreferenceStore()
        store.save(Preference(key="name", value="alice", confidence=0.5))
        store.save(Preference(key="name", value="bob", confidence=0.9))
        loaded = store.load("name")
        assert loaded is not None
        assert loaded.value == "bob"


class TestSqlitePreferenceStore:
    def test_save_and_load(self) -> None:
        store = SqlitePreferenceStore(":memory:")
        p = Preference(key="volume", value="high", confidence=0.7)
        store.save(p)
        loaded = store.load("volume")
        assert loaded is not None
        assert loaded.value == "high"
        store.close()

    def test_load_missing(self) -> None:
        store = SqlitePreferenceStore(":memory:")
        assert store.load("missing") is None
        store.close()

    def test_delete(self) -> None:
        store = SqlitePreferenceStore(":memory:")
        store.save(Preference(key="a", value="1"))
        assert store.delete("a") is True
        assert store.load("a") is None
        store.close()


class TestPreferenceTracker:
    def test_explicit_name(self) -> None:
        tracker = PreferenceTracker()
        prefs = tracker.process_user_text("My name is Alice")
        assert len(prefs) >= 1
        name_pref = tracker.get("name")
        assert name_pref is not None
        assert name_pref.value == "alice"
        assert name_pref.source == "explicit"

    def test_explicit_humour(self) -> None:
        tracker = PreferenceTracker()
        prefs = tracker.process_user_text("Be funny please")
        assert any(p.key == "humour" and p.value == "humorous" for p in prefs)

    def test_explicit_formality(self) -> None:
        tracker = PreferenceTracker()
        prefs = tracker.process_user_text("Be formal")
        assert any(p.key == "formality" and p.value == "formal" for p in prefs)

    def test_explicit_verbosity(self) -> None:
        tracker = PreferenceTracker()
        prefs = tracker.process_user_text("Be brief")
        assert any(p.key == "verbosity" and p.value == "brief" for p in prefs)

    def test_explicit_volume(self) -> None:
        tracker = PreferenceTracker()
        prefs = tracker.process_user_text("Too loud")
        assert any(p.key == "volume" and p.value == "low" for p in prefs)

    def test_confidence_increases(self) -> None:
        tracker = PreferenceTracker()
        tracker.process_user_text("Be funny")
        tracker.process_user_text("Be funny")
        tracker.process_user_text("Be funny")
        pref = tracker.get("humour")
        assert pref is not None
        assert pref.confidence > 0.5  # Increased from repeated observations

    def test_format_for_prompt(self) -> None:
        tracker = PreferenceTracker()
        tracker.process_user_text("My name is Bob")
        text = tracker.format_for_prompt()
        assert "name" in text
        assert "bob" in text

    def test_format_for_prompt_empty(self) -> None:
        tracker = PreferenceTracker()
        assert tracker.format_for_prompt() == ""

    def test_no_match(self) -> None:
        tracker = PreferenceTracker()
        prefs = tracker.process_user_text("The weather is nice today")
        assert len(prefs) == 0


class TestPreferencesConfig:
    def test_defaults(self) -> None:
        from robot.config import PreferencesConfig

        cfg = PreferencesConfig()
        assert cfg.enabled is True
        assert cfg.store == "memory"
        assert cfg.db_path == "~/.deskbot/preferences.db"

    def test_env_override(self) -> None:
        import os

        from robot.config import PreferencesConfig

        env = {
            "DESKBOT_PREFERENCES__ENABLED": "false",
            "DESKBOT_PREFERENCES__STORE": "sqlite",
            "DESKBOT_PREFERENCES__DB_PATH": "/tmp/test_prefs.db",
        }
        original = {}
        for key, value in env.items():
            original[key] = os.environ.get(key)
            os.environ[key] = value
        try:
            cfg = PreferencesConfig()
            assert cfg.enabled is False
            assert cfg.store == "sqlite"
            assert cfg.db_path == "/tmp/test_prefs.db"
        finally:
            for key, value in original.items():  # type: ignore[assignment]
                if value is None:
                    os.environ.pop(key, None)  # type: ignore[unreachable]
                else:
                    os.environ[key] = value


class TestConversationServiceWithPreferences:
    """Test that ConversationService correctly uses PreferenceTracker."""

    async def test_preference_tracker_processes_user_text(self) -> None:
        """When preference_tracker is set, process_user_text is called from _on_speech."""
        from robot.ai.conversation import ConversationManager
        from robot.ai.llm_mock import MockLLM
        from robot.ai.preferences import InMemoryPreferenceStore, PreferenceTracker
        from robot.behavior.state_machine import RobotState, StateMachine
        from robot.events.bus import InMemoryEventBus
        from robot.services.conversation_service import ConversationService
        from robot.speech.stt import MockSTT
        from robot.speech.tts import MockTTS

        bus = InMemoryEventBus()
        sm = StateMachine(bus=bus)
        # Transition BOOT -> IDLE -> LISTENING so _on_speech works.
        await sm.transition(RobotState.IDLE)
        await sm.transition(RobotState.LISTENING)

        llm = MockLLM()
        llm.register("hello", "Hi there!")
        tracker = PreferenceTracker(store=InMemoryPreferenceStore())
        cs = ConversationService(
            bus=bus,
            state_machine=sm,
            stt=MockSTT(),
            tts=MockTTS(),
            llm=llm,
            conversation=ConversationManager(llm=llm, system_prompt="test"),
            preference_tracker=tracker,
        )
        cs.attach()
        # Process a user utterance that contains a preference.
        from robot.events.events import SpeechRecognized

        await bus.publish(SpeechRecognized(text="My name is Bob", confidence=0.9))
        # Give the event handlers a chance to run.
        import asyncio

        for _ in range(10):
            await asyncio.sleep(0)

        pref = tracker.get("name")
        assert pref is not None
        assert pref.value == "bob"
        cs.detach()

    async def test_memory_context_includes_preferences(self) -> None:
        """_memory_context should include preferences when tracker is set."""
        from robot.ai.conversation import ConversationManager
        from robot.ai.llm_mock import MockLLM
        from robot.ai.memory import Memory
        from robot.ai.preferences import InMemoryPreferenceStore, PreferenceTracker
        from robot.behavior.state_machine import StateMachine
        from robot.events.bus import InMemoryEventBus
        from robot.services.conversation_service import ConversationService
        from robot.speech.stt import MockSTT
        from robot.speech.tts import MockTTS

        bus = InMemoryEventBus()
        sm = StateMachine(bus=bus)
        tracker = PreferenceTracker(store=InMemoryPreferenceStore())
        tracker.process_user_text("Be funny")

        cs = ConversationService(
            bus=bus,
            state_machine=sm,
            stt=MockSTT(),
            tts=MockTTS(),
            llm=MockLLM(),
            conversation=ConversationManager(llm=MockLLM(), system_prompt="test"),
            memory=Memory(capacity=100),
            preference_tracker=tracker,
        )
        context = cs._memory_context("hello")
        assert "humour" in context
        assert "humorous" in context

    def test_memory_context_without_preferences(self) -> None:
        """When preference_tracker is None, _memory_context still works."""
        from robot.ai.conversation import ConversationManager
        from robot.ai.llm_mock import MockLLM
        from robot.ai.memory import Memory
        from robot.behavior.state_machine import StateMachine
        from robot.events.bus import InMemoryEventBus
        from robot.services.conversation_service import ConversationService
        from robot.speech.stt import MockSTT
        from robot.speech.tts import MockTTS

        bus = InMemoryEventBus()
        sm = StateMachine(bus=bus)
        cs = ConversationService(
            bus=bus,
            state_machine=sm,
            stt=MockSTT(),
            tts=MockTTS(),
            llm=MockLLM(),
            conversation=ConversationManager(llm=MockLLM(), system_prompt="test"),
            memory=Memory(capacity=100),
        )
        context = cs._memory_context("hello")
        # No preferences, no memory matches - should be empty string.
        assert context == ""


class TestPreferenceTrackerExtended:
    """Extended tests for PreferenceTracker."""

    def test_process_name_preference(self) -> None:
        from robot.ai.preferences import InMemoryPreferenceStore, PreferenceTracker

        store = InMemoryPreferenceStore()
        tracker = PreferenceTracker(store=store)
        tracker.process_user_text("my name is Alice")
        prefs = tracker.get_all()
        # Should have extracted a name preference
        name_prefs = [p for p in prefs if p.key == "name"]
        assert len(name_prefs) >= 1

    def test_format_for_prompt_returns_string(self) -> None:
        from robot.ai.preferences import InMemoryPreferenceStore, PreferenceTracker

        store = InMemoryPreferenceStore()
        tracker = PreferenceTracker(store=store)
        result = tracker.format_for_prompt()
        assert isinstance(result, str)

    def test_format_for_prompt_with_preferences(self) -> None:
        from robot.ai.preferences import InMemoryPreferenceStore, PreferenceTracker

        store = InMemoryPreferenceStore()
        tracker = PreferenceTracker(store=store)
        tracker.process_user_text("my name is Bob")
        result = tracker.format_for_prompt()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_process_multiple_preferences(self) -> None:
        from robot.ai.preferences import InMemoryPreferenceStore, PreferenceTracker

        store = InMemoryPreferenceStore()
        tracker = PreferenceTracker(store=store)
        tracker.process_user_text("my name is Alice")
        tracker.process_user_text("I like quiet things")
        prefs = tracker.get_all()
        assert len(prefs) >= 1


class TestInMemoryPreferenceStoreExtra:
    """Tests for InMemoryPreferenceStore (extra cases)."""

    def test_store_and_retrieve(self) -> None:
        from robot.ai.preferences import InMemoryPreferenceStore, Preference

        store = InMemoryPreferenceStore()
        pref = Preference(key="name", value="Alice", confidence=0.9, source="explicit")
        store.save(pref)
        result = store.load("name")
        assert result is not None
        assert result.value == "Alice"

    def test_store_overwrite(self) -> None:
        from robot.ai.preferences import InMemoryPreferenceStore, Preference

        store = InMemoryPreferenceStore()
        store.save(Preference(key="name", value="Alice", confidence=0.5))
        store.save(Preference(key="name", value="Bob", confidence=0.8))
        result = store.load("name")
        assert result is not None
        assert result.value == "Bob"

    def test_load_missing_returns_none(self) -> None:
        from robot.ai.preferences import InMemoryPreferenceStore

        store = InMemoryPreferenceStore()
        assert store.load("missing") is None

    def test_list_all(self) -> None:
        from robot.ai.preferences import InMemoryPreferenceStore, Preference

        store = InMemoryPreferenceStore()
        store.save(Preference(key="name", value="Alice", confidence=0.9))
        store.save(Preference(key="humour", value="dry", confidence=0.7))
        all_prefs = store.load_all()
        assert len(all_prefs) == 2

    def test_delete(self) -> None:
        from robot.ai.preferences import InMemoryPreferenceStore, Preference

        store = InMemoryPreferenceStore()
        store.save(Preference(key="name", value="Alice", confidence=0.9))
        store.delete("name")
        assert store.load("name") is None
