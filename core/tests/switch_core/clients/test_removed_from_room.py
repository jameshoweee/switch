"""What an agent client does when it is taken out of a room while running.

Dropping the transport's subscription governs only what has not been read yet.
Everything already read is in the event buffer, which is keyed by agent and
knows nothing about who is in what: it holds each event for the retention
window and serves it to a long poll, to the notification stream, and to an SSE
reader resuming from an old cursor. So the removal has to reach the things
holding the events, not only the reader that would have fetched more.
"""

from __future__ import annotations

from types import SimpleNamespace

from switch_core.bridges.agent.protocol.connections import (
    PROTOCOL_VERSION,
    ClientDeclaration,
    ConnectionRegistry,
)
from switch_core.bridges.agent.protocol.event_buffer import EventBuffer
from switch_core.bridges.agent.protocol.types import AgentEvent, MessagePayload
from switch_core.clients.agent_client import AgentClient, RoomMeta
from switch_core.transport import InboundMembership, RoomRef

AGENT = "agent-1"
LEFT = "room-left"
KEPT = "room-kept"


def _message(room_id: str, body: str) -> AgentEvent:
    return AgentEvent(
        type="message",
        room_id=room_id,
        payload=MessagePayload(
            addressed=True,
            sender="@someone:test",
            sender_name="someone",
            message_id=f"$m-{body}",
            body=body,
            timestamp=0,
        ),
    )


def _leave(transport_room_id: str) -> InboundMembership:
    return InboundMembership(
        room_id=transport_room_id,
        event_id="$leave",
        sender="@agent:test",
        timestamp=0,
        state_key="@agent:test",
        membership="leave",
        display_name="agent",
    )


def _client(buffer: EventBuffer, connections: ConnectionRegistry) -> SimpleNamespace:
    """A minimal fake `self` for the unbound `AgentClient.on_removed`."""
    meta = {
        "!left:test": RoomMeta(
            room_id=LEFT,
            name="left",
            bridge_id=None,
            channel_type="channel_public",
        ),
    }

    async def _resolve_room_meta(matrix_room_id: str) -> RoomMeta | None:
        return meta.get(matrix_room_id)

    return SimpleNamespace(
        _event_buffer=buffer,
        _connections=connections,
        agent=SimpleNamespace(id=AGENT),
        _resolve_room_meta=_resolve_room_meta,
    )


async def _removed(stub: SimpleNamespace, transport_room_id: str) -> None:
    await AgentClient.on_removed(
        stub,  # type: ignore[arg-type]
        RoomRef(room_id=transport_room_id),
        _leave(transport_room_id),
    )


async def test_the_rooms_retained_events_are_forgotten() -> None:
    buffer = EventBuffer()
    buffer.enqueue(AGENT, LEFT, _message(LEFT, "said in the old room"))
    buffer.enqueue(AGENT, KEPT, _message(KEPT, "said in this one"))

    await _removed(_client(buffer, ConnectionRegistry()), "!left:test")

    # Read at the level every reader is built on, filter or no filter: this is
    # what closes it for the stream, which has no membership of its own to
    # apply and would otherwise replay the room on every resume.
    assert [item.room_id for item in buffer.read_from(AGENT, 0)] == [KEPT]


async def test_the_rooms_claim_is_released() -> None:
    connections = ConnectionRegistry()
    conn = connections.open(
        agent_id=AGENT,
        connection_id="c1",
        scope="single",
        delivery_filter="all",
        spawn_capable=False,
        cursor=0,
        declaration=ClientDeclaration(speaks=PROTOCOL_VERSION),
    )
    connections.claim_room(conn, LEFT)

    await _removed(_client(EventBuffer(), connections), "!left:test")

    assert conn.rooms == set()


async def test_a_room_that_cannot_be_resolved_drops_nothing() -> None:
    """Rather than guessing which room was meant and emptying the wrong one."""
    buffer = EventBuffer()
    buffer.enqueue(AGENT, LEFT, _message(LEFT, "still here"))

    await _removed(_client(buffer, ConnectionRegistry()), "!unknown:test")

    assert [item.room_id for item in buffer.read_from(AGENT, 0)] == [LEFT]
