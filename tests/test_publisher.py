"""Tests for the UDP Publisher's NaN/Inf guard."""

import json

from yfips.detection import Publisher


class _StubSock:
    def __init__(self):
        self.sent = []

    def sendto(self, data, addr):
        self.sent.append(data)


def _make_pub():
    pub = Publisher({"enabled": True, "host": "127.0.0.1", "port": 65500})
    pub.sock = _StubSock()
    return pub, pub.sock.sent


def test_send_passes_finite_payload():
    pub, sent = _make_pub()
    pub.send({"id": 1, "x": 1.0, "y": 2.0, "yaw": 0.5, "t": 0.0})
    assert len(sent) == 1
    payload = json.loads(sent[0])
    assert payload["x"] == 1.0
    assert payload["y"] == 2.0


def test_send_drops_nan():
    pub, sent = _make_pub()
    pub.send({"id": 1, "x": float("nan"), "y": 0.0, "yaw": 0.0, "t": 0.0})
    assert sent == []


def test_send_drops_inf():
    pub, sent = _make_pub()
    pub.send({"id": 1, "x": 0.0, "y": float("inf"), "yaw": 0.0, "t": 0.0})
    assert sent == []


def test_send_drops_negative_inf_yaw():
    pub, sent = _make_pub()
    pub.send({"id": 1, "x": 0.0, "y": 0.0, "yaw": float("-inf"), "t": 0.0})
    assert sent == []


def test_disabled_publisher_does_not_send():
    pub = Publisher({"enabled": False})
    # No exception; sock is None when disabled.
    pub.send({"id": 1, "x": 1.0, "y": 2.0, "yaw": 0.5, "t": 0.0})
