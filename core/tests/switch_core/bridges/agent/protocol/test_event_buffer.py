"""Unit tests for the EventBuffer notification fan-out (CHOO-889).

The auto_session watcher consumes a separate, agent-scoped notification stream.
The critical invariant: fanning an event out to that stream must NOT remove it
from the per-room queue a live session poller drains — otherwise the watcher
would steal events from connected rooms.
"""

from __future__ import annotations

import asyncio

from switch_core.bridges.agent.protocol.event_buffer import EventBuffer, fixed_rooms
from switch_core.bridges.agent.protocol.types import (
    AgentEvent,
    MessagePayload,
    RoomJoinPayload,
    TaskDelegatePayload,
)

AGENT = "agent-1"
ROOM = "room-1"


def _message(addressed: bool, room_id: str = ROOM) -> AgentEvent:
    return AgentEvent(
        type="message",
        room_id=room_id,
        payload=MessagePayload(
            addressed=addressed,
            sender="@u:s",
            sender_name="u",
            message_id="$m",
            body="hi",
            timestamp=0,
        ),
    )


def _room_join(listening: bool) -> AgentEvent:
    return AgentEvent(
        type="room_join",
        room_id=ROOM,
        payload=RoomJoinPayload(
            member="@u:s", member_name="u", timestamp=0, listening=listening
        ),
    )


def _task_delegate() -> AgentEvent:
    return AgentEvent(
        type="task_delegate",
        room_id=ROOM,
        payload=TaskDelegatePayload(
            task_id="t1",
            requester_agent_id="r",
            performer_agent_id=AGENT,
            summary="s",
            description="d",
        ),
    )


async def test_addressed_message_fans_out_without_draining_room_queue() -> None:
    q = EventBuffer()
    q.enqueue(AGENT, ROOM, _message(addressed=True))

    # The notification stream sees it...
    notifs = await q.poll_notifications(AGENT, timeout=0, rooms=fixed_rooms({ROOM}))
    assert len(notifs) == 1
    assert notifs[0].type == "message"

    # ...and it is STILL waiting in the per-room queue for the session poller.
    room_events = await q.poll_room(AGENT, ROOM, timeout=0)
    assert len(room_events) == 1


async def test_unaddressed_message_does_not_fan_out() -> None:
    q = EventBuffer()
    q.enqueue(AGENT, ROOM, _message(addressed=False))

    assert await q.poll_notifications(AGENT, timeout=0, rooms=fixed_rooms({ROOM})) == []
    # Still queued per-room (unaddressed chatter is delivered there, just not
    # surfaced as a notification).
    assert len(await q.poll_room(AGENT, ROOM, timeout=0)) == 1


async def test_task_event_fans_out() -> None:
    q = EventBuffer()
    q.enqueue(AGENT, ROOM, _task_delegate())
    notifs = await q.poll_notifications(AGENT, timeout=0, rooms=fixed_rooms({ROOM}))
    assert len(notifs) == 1
    assert notifs[0].type == "task_delegate"


async def test_room_join_fans_out_only_when_listening() -> None:
    q = EventBuffer()
    q.enqueue(AGENT, ROOM, _room_join(listening=False))
    assert await q.poll_notifications(AGENT, timeout=0, rooms=fixed_rooms({ROOM})) == []

    q.enqueue(AGENT, ROOM, _room_join(listening=True))
    notifs = await q.poll_notifications(AGENT, timeout=0, rooms=fixed_rooms({ROOM}))
    assert len(notifs) == 1
    assert notifs[0].type == "room_join"


async def test_polling_everything_can_be_limited_to_the_rooms_given() -> None:
    """The buffer is keyed by agent and knows nothing about who is in what.

    An event queued while the agent was a member stays queued after it is
    removed, so the caller passes the rooms it is in now and that is what
    keeps the event from being handed over.
    """
    q = EventBuffer()
    q.enqueue(AGENT, ROOM, _message(addressed=False))
    q.enqueue(AGENT, "room-2", _message(addressed=False, room_id="room-2"))

    polled = await q.poll(AGENT, timeout=0, rooms=fixed_rooms({"room-2"}))

    assert [event.room_id for event in polled] == ["room-2"]


async def test_polling_with_no_rooms_at_all_returns_nothing() -> None:
    """An agent in no rooms is not a caller asking for every room."""
    q = EventBuffer()
    q.enqueue(AGENT, ROOM, _message(addressed=False))

    assert await q.poll(AGENT, timeout=0, rooms=fixed_rooms(set())) == []


async def test_notifications_are_limited_to_the_rooms_given_too() -> None:
    """The notification stream is the one carrying addressed messages.

    Filtering the all-rooms poll and not this one would leave the leak open on
    the more sensitive of the two: an agent removed from a room would stop
    seeing its chatter and go on being handed everything said *to* it there.
    """
    q = EventBuffer()
    q.enqueue(AGENT, ROOM, _message(addressed=True))
    q.enqueue(AGENT, "room-2", _message(addressed=True, room_id="room-2"))

    polled = await q.poll_notifications(AGENT, timeout=0, rooms=fixed_rooms({"room-2"}))

    assert [event.room_id for event in polled] == ["room-2"]


async def test_the_room_set_is_asked_for_after_the_wait_not_before() -> None:
    """A poll parked for its timeout must not answer from a stale membership.

    The events that wake it are, by definition, events that arrived during the
    wait — including the first ones from a room the agent was added to while it
    was parked.
    """
    q = EventBuffer()
    asked: list[int] = []

    async def _rooms() -> set[str]:
        asked.append(len(asked))
        # Not a member of anything on the way in; a member by the time the
        # wait ends.
        return set() if len(asked) == 1 else {"room-2"}

    async def _arrive() -> None:
        q.enqueue(AGENT, "room-2", _message(addressed=False, room_id="room-2"))

    polling = asyncio.create_task(q.poll(AGENT, timeout=5, rooms=_rooms))
    for _ in range(10):
        await asyncio.sleep(0)
    await _arrive()

    assert [event.room_id for event in await polling] == ["room-2"]
    assert len(asked) == 2


async def test_dropping_a_room_forgets_what_it_still_held() -> None:
    """What the transport stops reading, the buffer has to stop holding.

    Dropping the subscription only governs what has not been read yet. An
    event already in here is served to a long poll, to the notification stream
    and to an SSE reader resuming from an old cursor for the whole retention
    window.
    """
    q = EventBuffer()
    q.enqueue(AGENT, ROOM, _message(addressed=True))
    q.enqueue(AGENT, "room-2", _message(addressed=True, room_id="room-2"))

    q.drop_room(AGENT, ROOM)

    # Gone from the low-level read every reader is built on, filter or no
    # filter — which is what closes it for the stream, the one reader with no
    # membership of its own to apply.
    assert [item.room_id for item in q.read_from(AGENT, 0)] == ["room-2"]


async def test_dropping_a_room_does_not_report_a_gap() -> None:
    """A reader that skips the dropped events has missed nothing it was owed.

    Saying otherwise would send an agent off to re-read the context of a room
    it is no longer in.
    """
    q = EventBuffer()
    q.enqueue(AGENT, ROOM, _message(addressed=True))
    q.enqueue(AGENT, "room-2", _message(addressed=True, room_id="room-2"))

    q.drop_room(AGENT, ROOM)

    assert not q.has_gap_before(AGENT, 0)
    assert [item.seq for item in q.read_from(AGENT, 0)] == [2]


async def test_a_filtered_out_event_does_not_strand_the_poll_cursor() -> None:
    """The all-rooms cursor advances past what the filter excluded.

    Left behind them it parks below the head for good; retention eventually
    trims them, and the next read raises `CursorExpiredError` and logs that the
    poller missed events. Nothing was missed — they were filtered.
    """
    q = EventBuffer()
    q.enqueue(AGENT, ROOM, _message(addressed=False))
    q.enqueue(AGENT, "room-2", _message(addressed=False, room_id="room-2"))
    q.enqueue(AGENT, "room-2", _message(addressed=False, room_id="room-2"))

    # Removed from both: everything retained is filtered out.
    assert await q.poll(AGENT, timeout=0, rooms=fixed_rooms(set())) == []
    assert q._cursors[AGENT]["legacy:all"] == 3


async def test_remove_clears_notification_queue() -> None:
    q = EventBuffer()
    q.enqueue(AGENT, ROOM, _message(addressed=True))
    q.remove(AGENT)
    assert await q.poll_notifications(AGENT, timeout=0, rooms=fixed_rooms({ROOM})) == []
