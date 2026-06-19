from bet import place_bet


class DummyResponse:
    def __init__(self, status_code=201, body=None, text=""):
        self.status_code = status_code
        self._body = body or {}
        self.text = text

    def json(self):
        return self._body


class DummySession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.posts = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.posts.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return self.responses.pop(0)


def test_events_order_payload(monkeypatch):
    monkeypatch.setattr(place_bet, "auth_headers", lambda *args, **kwargs: {"auth": "ok"})
    monkeypatch.setattr(place_bet, "KALSHI_ORDER_ENDPOINT", "events")
    session = DummySession([
        DummyResponse(body={
            "order_id": "ord_1",
            "fill_count": "2.00",
            "average_fill_price": "0.53",
        })
    ])

    order, api = place_bet._post_kalshi_order(
        session, "https://kalshi.test/trade-api/v2", "key", object(), "TICKER", 2, 53
    )

    assert api == "events"
    assert order["order_id"] == "ord_1"
    assert session.posts[0]["url"].endswith("/portfolio/events/orders")
    assert session.posts[0]["json"]["side"] == "bid"
    assert session.posts[0]["json"]["price"] == "0.5300"
    assert session.posts[0]["json"]["count"] == "2.00"


def test_events_shape_rejection_does_not_fall_back_to_legacy(monkeypatch):
    monkeypatch.setattr(place_bet, "auth_headers", lambda *args, **kwargs: {"auth": "ok"})
    monkeypatch.setattr(place_bet, "KALSHI_ORDER_ENDPOINT", "events")
    session = DummySession([
        DummyResponse(status_code=422, text="invalid price schema"),
    ])

    try:
        place_bet._post_kalshi_order(
            session, "https://kalshi.test/trade-api/v2", "key", object(), "TICKER", 1, 51
        )
    except place_bet.PlaceBetError as exc:
        assert "order rejected (422): invalid price schema" in str(exc)
    else:
        raise AssertionError("expected PlaceBetError")

    assert session.posts[0]["url"].endswith("/portfolio/events/orders")
    assert len(session.posts) == 1
