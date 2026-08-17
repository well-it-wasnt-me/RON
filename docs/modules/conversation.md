# Conversation and AI

The AI subsystem combines an LLM, conversation state, persistent storage,
short-term memory, optional vector-memory support, and tool calling.

## LLM providers

Current implementations include:

- `MockLLM`
- `OpenAILLM`
- `OllamaLLM`

`StreamingLLM` is the provider protocol used for token streaming.

### OpenAI

Configure:

```env
DESKBOT_LLM__PROVIDER=openai
DESKBOT_LLM__MODEL=gpt-4o-mini
DESKBOT_LLM__API_KEY=...
```

### Ollama

Configure the provider and its base URL/model according to the target Ollama
installation.

```env
DESKBOT_LLM__PROVIDER=ollama
DESKBOT_LLM__MODEL=llama3
DESKBOT_LLM__BASE_URL=http://localhost:11434
```

## Conversation manager

`ConversationManager` owns the active conversation and sends message history
to the selected LLM.

Messages are bounded by the conversation history limit and can be persisted
through a `ConversationStore`.

## Persistence

The store abstraction supports:

- `InMemoryStore`
- `SqliteConversationStore`

Configure:

```env
DESKBOT_CONVERSATION__STORE=sqlite
DESKBOT_CONVERSATION__DB_PATH=~/.deskbot/conversations.db
DESKBOT_CONVERSATION__CONVERSATION_ID=default
```

The REST API can list, retrieve, and delete persisted conversations.

## Memory

`Memory` is a bounded ring buffer of `MemoryEntry` values with importance and
tags. Relevant recalled context can be injected into LLM prompts.

This is not the same thing as durable conversation history.

Vector-memory support exists in the package but should be treated as an
extension point rather than assuming a configured semantic-search backend is
always active.

## Tool calling

The tool subsystem contains:

- tool schemas
- a registry
- an executor

When enabled, tool definitions can be supplied to compatible LLM providers and
tool calls can be dispatched through the executor.

```env
DESKBOT_TOOLS__ENABLED=true
```

## Streaming

Streaming providers emit `LLMTokenReceived` events. `FaceOrchestrator` consumes
those events to coordinate thinking/speaking facial animation while a response
is generated.
