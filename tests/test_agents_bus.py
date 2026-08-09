"""The message bus: routing, ordering, immutability, and typo resistance."""

from __future__ import annotations

import pytest

from pharmadt.agents.bus import TOPIC_CONTRACTS, Message, MessageBus, Topic

# The eight labelled edges of the agent architecture diagram.
EXPECTED_TOPICS = {
    "forecast.data",
    "shortage.alert",
    "demand.hotspot",
    "counterfeit.flag",
    "replenishment.order",
    "redistribution.request",
    "route.plan",
    "ledger.event",
}


@pytest.fixture
def bus() -> MessageBus:
    return MessageBus()


# ── Topics mirror the architecture diagram ────────────────────────────


def test_topics_are_exactly_the_diagram_edges() -> None:
    """If these drift, the report's diagram stops describing the code."""
    assert {t.value for t in Topic} == EXPECTED_TOPICS


def test_every_topic_declares_a_publisher_and_subscribers() -> None:
    assert set(TOPIC_CONTRACTS) == set(Topic)
    for topic, (publisher, subscribers) in TOPIC_CONTRACTS.items():
        assert publisher, topic
        assert subscribers, topic


def test_an_unknown_topic_is_rejected(bus: MessageBus) -> None:
    """A mistyped topic would otherwise reach nobody and raise nothing."""
    with pytest.raises(ValueError, match="unknown topic"):
        bus.publish("replenishment.oder", {})

    with pytest.raises(ValueError, match="unknown topic"):
        bus.subscribe("not.a.topic", lambda m: None)


# ── Delivery ──────────────────────────────────────────────────────────


def test_a_subscriber_receives_what_was_published(bus: MessageBus) -> None:
    received: list[Message] = []
    bus.subscribe(Topic.SHORTAGE_ALERT, received.append)

    delivered = bus.publish(
        Topic.SHORTAGE_ALERT, {"node_id": "NODE-PH-01"}, sender="DemandAgent", sim_day=7
    )

    assert delivered == 1
    assert len(received) == 1
    assert received[0].topic is Topic.SHORTAGE_ALERT
    assert received[0].sender == "DemandAgent"
    assert received[0].sim_day == 7
    assert received[0].payload["node_id"] == "NODE-PH-01"


def test_publishing_with_no_subscribers_is_harmless(bus: MessageBus) -> None:
    """The Route Agent may not exist yet when the Inventory Agent starts."""
    assert bus.publish(Topic.ROUTE_PLAN, {}) == 0


def test_every_subscriber_of_a_topic_is_reached(bus: MessageBus) -> None:
    """counterfeit.flag fans out to the Expiry Agent and the dashboard."""
    a, b = [], []
    bus.subscribe(Topic.COUNTERFEIT_FLAG, a.append)
    bus.subscribe(Topic.COUNTERFEIT_FLAG, b.append)

    assert bus.publish(Topic.COUNTERFEIT_FLAG, {"batch_id": "B1"}) == 2
    assert len(a) == len(b) == 1


def test_handlers_fire_in_registration_order(bus: MessageBus) -> None:
    """Set iteration would reorder between runs and break run reproducibility."""
    order: list[str] = []
    for name in ("first", "second", "third"):
        bus.subscribe(Topic.FORECAST_DATA, lambda _m, n=name: order.append(n))

    bus.publish(Topic.FORECAST_DATA, {})
    assert order == ["first", "second", "third"]


def test_a_message_only_reaches_its_own_topic(bus: MessageBus) -> None:
    forecasts, routes = [], []
    bus.subscribe(Topic.FORECAST_DATA, forecasts.append)
    bus.subscribe(Topic.ROUTE_PLAN, routes.append)

    bus.publish(Topic.FORECAST_DATA, {})
    assert len(forecasts) == 1
    assert routes == []


# ── Subscription management ───────────────────────────────────────────


def test_subscribing_the_same_handler_twice_does_not_double_deliver(
    bus: MessageBus,
) -> None:
    """A doubly-registered ordering agent would order everything twice."""
    received = []
    bus.subscribe(Topic.REPLENISHMENT_ORDER, received.append)
    bus.subscribe(Topic.REPLENISHMENT_ORDER, received.append)

    bus.publish(Topic.REPLENISHMENT_ORDER, {})
    assert len(received) == 1


def test_unsubscribing_stops_delivery(bus: MessageBus) -> None:
    received = []
    bus.subscribe(Topic.ROUTE_PLAN, received.append)
    bus.unsubscribe(Topic.ROUTE_PLAN, received.append)
    bus.publish(Topic.ROUTE_PLAN, {})
    assert received == []


def test_unsubscribing_something_never_subscribed_is_a_noop(bus: MessageBus) -> None:
    bus.unsubscribe(Topic.ROUTE_PLAN, lambda m: None)


def test_a_handler_may_subscribe_while_being_delivered_to(bus: MessageBus) -> None:
    """Iterating the live list would raise mid-delivery."""
    late = []

    def on_flag(_message: Message) -> None:
        bus.subscribe(Topic.COUNTERFEIT_FLAG, late.append)

    bus.subscribe(Topic.COUNTERFEIT_FLAG, on_flag)
    bus.publish(Topic.COUNTERFEIT_FLAG, {})  # must not raise
    assert late == []

    bus.publish(Topic.COUNTERFEIT_FLAG, {})
    assert len(late) == 1


def test_clear_removes_subscriptions_and_history(bus: MessageBus) -> None:
    bus.subscribe(Topic.ROUTE_PLAN, lambda m: None)
    bus.publish(Topic.ROUTE_PLAN, {})
    bus.clear()

    assert bus.subscribers(Topic.ROUTE_PLAN) == ()
    assert len(bus) == 0
    assert bus.published_count == 0


# ── Immutability ──────────────────────────────────────────────────────


def test_a_message_cannot_be_reassigned() -> None:
    import dataclasses

    message = Message(topic=Topic.ROUTE_PLAN, sender="RouteAgent", sim_day=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        message.sim_day = 2  # type: ignore[misc]


def test_one_subscriber_cannot_rewrite_what_the_next_receives(bus: MessageBus) -> None:
    """Otherwise registration order becomes hidden business logic."""
    seen: list[int] = []

    def greedy(message: Message) -> None:
        with pytest.raises(TypeError):
            message.payload["quantity"] = 0  # type: ignore[index]

    bus.subscribe(Topic.REPLENISHMENT_ORDER, greedy)
    bus.subscribe(Topic.REPLENISHMENT_ORDER, lambda m: seen.append(m.payload["quantity"]))

    bus.publish(Topic.REPLENISHMENT_ORDER, {"quantity": 500})
    assert seen == [500]


def test_mutating_the_source_dict_afterwards_does_not_change_the_message(
    bus: MessageBus,
) -> None:
    payload = {"quantity": 100}
    received = []
    bus.subscribe(Topic.REPLENISHMENT_ORDER, received.append)

    bus.publish(Topic.REPLENISHMENT_ORDER, payload)
    payload["quantity"] = 999

    assert received[0].payload["quantity"] == 100


# ── Failure behaviour ─────────────────────────────────────────────────


def test_a_failing_handler_propagates(bus: MessageBus) -> None:
    """Swallowing it would let the run finish and report KPIs that are wrong."""

    def broken(_message: Message) -> None:
        raise RuntimeError("agent blew up")

    bus.subscribe(Topic.SHORTAGE_ALERT, broken)
    with pytest.raises(RuntimeError, match="agent blew up"):
        bus.publish(Topic.SHORTAGE_ALERT, {})


# ── History ───────────────────────────────────────────────────────────


def test_history_records_traffic_for_inspection(bus: MessageBus) -> None:
    bus.publish(Topic.FORECAST_DATA, {"a": 1})
    bus.publish(Topic.ROUTE_PLAN, {"b": 2})

    assert len(bus.history()) == 2
    assert [m.topic for m in bus.history(Topic.ROUTE_PLAN)] == [Topic.ROUTE_PLAN]
    assert bus.published_count == 2


def test_history_is_bounded() -> None:
    """A 365-day run must not accumulate messages without limit."""
    bus = MessageBus(history_limit=10)
    for day in range(50):
        bus.publish(Topic.FORECAST_DATA, {"day": day}, sim_day=day)

    assert len(bus) == 10
    assert bus.published_count == 50
    assert bus.history()[-1].payload["day"] == 49


def test_delivery_counters_track_fan_out(bus: MessageBus) -> None:
    bus.subscribe(Topic.COUNTERFEIT_FLAG, lambda m: None)
    bus.subscribe(Topic.COUNTERFEIT_FLAG, lambda m: None)
    bus.publish(Topic.COUNTERFEIT_FLAG, {})

    assert bus.published_count == 1
    assert bus.delivered_count == 2
