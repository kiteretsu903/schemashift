"""Keep the public verification suite independent of provider access."""

import socket

import pytest


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def denied(*_args, **_kwargs):
        raise RuntimeError("The public test suite must not make network requests")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
