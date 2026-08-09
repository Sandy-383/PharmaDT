"""In-process publish/subscribe bus connecting the five agents.

Topics correspond exactly to the labelled edges in the agent architecture
diagram, so the code and the diagram can be checked against each other rather
than drifting apart.

The bus is in-process on purpose. Redis pub/sub would be the answer if the
agents were ever split across processes, but the simulation is single-threaded
and an in-memory registry is both sufficient and far easier to debug — a
misrouted message is a stack frame away rather than in another service's log.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

logger = logging.getLogger(__name__)

#: How many messages to retain for inspection. The Stage 14 dashboard renders
#: recent agent traffic; keeping every message from a 365-day run would grow
#: without bound for no benefit.
HISTORY_LIMIT = 2000


class Topic(StrEnum):
    """The labelled edges of the agent architecture diagram.

    An enum rather than bare strings because a mistyped topic is otherwise a
    silent failure: ``publish("replenishment.oder", ...)`` reaches nobody,
    raises nothing, and shows up only as an agent that mysteriously never acts.
    """

    FORECAST_DATA = "forecast.data"
    SHORTAGE_ALERT = "shortage.alert"
    DEMAND_HOTSPOT = "demand.hotspot"
    COUNTERFEIT_FLAG = "counterfeit.flag"
    REPLENISHMENT_ORDER = "replenishment.order"
    REDISTRIBUTION_REQUEST = "redistribution.request"
    ROUTE_PLAN = "route.plan"
    LEDGER_EVENT = "ledger.event"


#: Who is expected on each end of every edge. Documentation that a test can
#: check, so the diagram in the report stays honest about the code.
TOPIC_CONTRACTS: Mapping[Topic, tuple[str, tuple[str, ...]]] = MappingProxyType(
    {
        Topic.FORECAST_DATA: ("DemandAgent", ("InventoryAgent",)),
        Topic.SHORTAGE_ALERT: ("DemandAgent", ("InventoryAgent",)),
        Topic.DEMAND_HOTSPOT: ("DemandAgent", ("ExpiryAgent",)),
        Topic.COUNTERFEIT_FLAG: ("AnomalyAgent", ("ExpiryAgent", "Dashboard")),
        Topic.REPLENISHMENT_ORDER: ("InventoryAgent", ("RouteAgent",)),
        Topic.REDISTRIBUTION_REQUEST: ("ExpiryAgent", ("RouteAgent",)),
        Topic.ROUTE_PLAN: ("RouteAgent", ("DigitalTwin",)),
        Topic.LEDGER_EVENT: ("Ledger", ("AnomalyAgent",)),
    }
)


@dataclass(frozen=True, slots=True)
class Message:
    """One message on the bus.

    Frozen because a single publish fans out to several handlers; a mutable
    payload would let the first subscriber change what the rest receive, which
    turns subscriber registration order into hidden business logic.
    """

    topic: Topic
    sender: str
    sim_day: int
    payload: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        # Freeze the payload too, so `msg.payload["x"] = 1` cannot succeed.
        if not isinstance(self.payload, MappingProxyType):
            object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))


Handler = Callable[[Message], None]


class MessageBus:
    """Synchronous in-process pub/sub."""

    def __init__(self, history_limit: int = HISTORY_LIMIT) -> None:
        # Lists, not sets: handlers fire in registration order, and set
        # iteration over objects would reorder between runs and break the
        # simulation's byte-identical guarantee.
        self._subscribers: dict[Topic, list[Handler]] = {}
        self._history: deque[Message] = deque(maxlen=history_limit)
        self.published_count = 0
        self.delivered_count = 0

    # ── Subscription ──────────────────────────────────────────────────

    def subscribe(self, topic: Topic, handler: Handler) -> None:
        """Register ``handler`` for ``topic``. Duplicate registrations are ignored."""
        topic = self._validate(topic)
        handlers = self._subscribers.setdefault(topic, [])
        if handler in handlers:
            # Subscribing the same handler twice would double every message it
            # sees, which for an ordering agent means ordering everything twice.
            logger.debug("handler already subscribed to %s", topic)
            return
        handlers.append(handler)

    def unsubscribe(self, topic: Topic, handler: Handler) -> None:
        handlers = self._subscribers.get(self._validate(topic), [])
        if handler in handlers:
            handlers.remove(handler)

    def subscribers(self, topic: Topic) -> tuple[Handler, ...]:
        return tuple(self._subscribers.get(self._validate(topic), ()))

    def clear(self) -> None:
        """Drop all subscriptions and history."""
        self._subscribers.clear()
        self._history.clear()
        self.published_count = 0
        self.delivered_count = 0

    # ── Publication ───────────────────────────────────────────────────

    def publish(
        self,
        topic: Topic,
        payload: Mapping[str, Any] | None = None,
        *,
        sender: str = "",
        sim_day: int = 0,
    ) -> int:
        """Deliver a message to every subscriber. Returns the number reached.

        Delivery is synchronous and in registration order. A handler that
        raises propagates: a research simulation that silently swallowed an
        agent failure would carry on producing KPIs that look fine and are
        quietly wrong.
        """
        message = Message(
            topic=self._validate(topic), sender=sender, sim_day=sim_day,
            payload=payload or {},
        )
        return self.publish_message(message)

    def publish_message(self, message: Message) -> int:
        self._history.append(message)
        self.published_count += 1

        # Copy before iterating: a handler is allowed to subscribe or
        # unsubscribe in response to a message, which would otherwise mutate
        # the list mid-iteration.
        handlers = list(self._subscribers.get(message.topic, ()))
        for handler in handlers:
            handler(message)

        self.delivered_count += len(handlers)
        if not handlers:
            logger.debug("no subscribers for %s", message.topic)
        return len(handlers)

    # ── Inspection ────────────────────────────────────────────────────

    def history(self, topic: Topic | None = None) -> list[Message]:
        if topic is None:
            return list(self._history)
        topic = self._validate(topic)
        return [m for m in self._history if m.topic is topic]

    def __iter__(self) -> Iterator[Message]:
        return iter(list(self._history))

    def __len__(self) -> int:
        return len(self._history)

    @staticmethod
    def _validate(topic: Topic) -> Topic:
        """Reject anything that is not a declared edge of the architecture."""
        try:
            return Topic(topic)
        except ValueError as exc:
            known = ", ".join(sorted(t.value for t in Topic))
            raise ValueError(f"unknown topic {topic!r}; expected one of: {known}") from exc
