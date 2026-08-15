"""Conversation-history REST API endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from robot.api.schemas import OkResponse

router = APIRouter()

if TYPE_CHECKING:
    from robot.ai.conversation import ConversationManager


class ConversationSummary(BaseModel):
    """Metadata for one saved conversation."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "abc123",
                    "created_at": "2026-08-12T10:00:00Z",
                    "system_prompt": "You are a helpful desktop robot.",
                    "message_count": 8,
                }
            ]
        }
    }

    id: str
    created_at: str
    system_prompt: str
    message_count: int


class MessageItem(BaseModel):
    """A single conversation message."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"role": "user", "content": "Hello! Who are you?"},
                {"role": "assistant", "content": "I'm DeskBot, your desktop companion!"},
            ]
        }
    }

    role: str
    content: str


class ConversationDetail(BaseModel):
    """A conversation and its messages."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "abc123",
                    "system_prompt": "You are a helpful desktop robot.",
                    "messages": [
                        {"role": "user", "content": "Hello! Who are you?"},
                        {"role": "assistant", "content": "I'm DeskBot, your desktop companion!"},
                    ],
                }
            ]
        }
    }

    id: str
    system_prompt: str
    messages: list[MessageItem] = Field(default_factory=list)


def _manager(request: Request) -> ConversationManager:
    bridge = getattr(request.app.state, "bridge", None)
    if bridge is None or bridge.conversation is None:
        raise HTTPException(status_code=503, detail="Conversation service is not available")
    return cast("ConversationManager", bridge.conversation.conversation)


@router.get(
    "/conversations", response_model=list[ConversationSummary], summary="List conversations"
)
async def list_conversations(request: Request) -> list[ConversationSummary]:
    """List durable conversations, newest first.

    With the default in-memory configuration, the active conversation is
    returned as a single ephemeral entry.
    """
    manager = _manager(request)
    if manager.store is None:
        return [
            ConversationSummary(
                id=manager.conversation_id,
                created_at="",
                system_prompt=manager.current.system_prompt,
                message_count=len(manager.current.messages),
            )
        ]
    return [
        ConversationSummary(
            id=meta.id,
            created_at=meta.created_at.isoformat(),
            system_prompt=meta.system_prompt,
            message_count=meta.message_count,
        )
        for meta in await manager.store.list_conversations()
    ]


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationDetail,
    summary="Get conversation",
)
async def get_conversation(request: Request, conversation_id: str) -> ConversationDetail:
    """Get a conversation, including ordered messages."""
    manager = _manager(request)
    if conversation_id == manager.conversation_id:
        conversation = manager.current
        return ConversationDetail(
            id=conversation_id,
            system_prompt=conversation.system_prompt,
            messages=[
                MessageItem(role=item.role.value, content=item.content)
                for item in conversation.messages
            ],
        )
    if manager.store is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = await manager.store.load(conversation_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    metadata = next(
        (meta for meta in await manager.store.list_conversations() if meta.id == conversation_id),
        None,
    )
    if metadata is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationDetail(
        id=conversation_id,
        system_prompt=metadata.system_prompt,
        messages=[MessageItem(role=role, content=content) for role, content in messages],
    )


@router.delete(
    "/conversations/{conversation_id}",
    summary="Delete conversation",
    response_model=OkResponse,
)
async def delete_conversation(request: Request, conversation_id: str) -> OkResponse:
    """Delete a saved conversation; deleting the active one resets it."""
    manager = _manager(request)
    if conversation_id == manager.conversation_id:
        manager.reset()
    if manager.store is None:
        if conversation_id != manager.conversation_id:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return OkResponse.model_validate({"status": "ok", "id": conversation_id})

    deleted = await manager.store.delete(conversation_id)
    if not deleted and conversation_id != manager.conversation_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return OkResponse.model_validate({"status": "ok", "id": conversation_id})
