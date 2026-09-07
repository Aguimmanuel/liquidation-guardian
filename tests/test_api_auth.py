"""B4: optional RS_API_TOKEN gate + same-origin CORS default.

The server is booted WITHOUT the lifespan/autopilot (plain TestClient, no
context manager) so no market network is touched; the sim backend answers
offline requests directly.
"""
from __future__ import annotations

import os

from fastapi.testclient import TestClient

import app.main as main_app


def _client(api_token: str, cors_origins: str = "") -> TestClient:
    old_token = os.environ.get("RS_API_TOKEN")
    old_cors = os.environ.get("RS_CORS_ORIGINS")
    os.environ["RS_API_TOKEN"] = api_token
    os.environ["RS_CORS_ORIGINS"] = cors_origins
    try:
        app = main_app.create_app()
    finally:
        if old_token is None:
            os.environ.pop("RS_API_TOKEN", None)
        else:
            os.environ["RS_API_TOKEN"] = old_token
        if old_cors is None:
            os.environ.pop("RS_CORS_ORIGINS", None)
        else:
            os.environ["RS_CORS_ORIGINS"] = old_cors
    # no `with`: the lifespan (background autopilot) must not start in tests
    return TestClient(app)


def test_state_change_open_when_no_token_configured():
    client = _client("")
    r = client.post(
        "/api/funds/transfer",
        json={"direction": "to_futures", "amount_usdt": 10.0},
    )
    assert r.status_code != 401, "no token configured -> API stays open"


def test_state_change_rejected_without_token():
    client = _client(api_token="sekrit")
    r = client.post(
        "/api/funds/transfer",
        json={"direction": "to_futures", "amount_usdt": 10.0},
    )
    assert r.status_code == 401
    assert "token" in r.json()["message"].lower()


def test_state_change_accepted_with_bearer_token():
    client = _client(api_token="sekrit")
    r = client.post(
        "/api/funds/transfer",
        headers={"Authorization": "Bearer sekrit"},
        json={"direction": "to_futures", "amount_usdt": 10.0},
    )
    assert r.status_code == 200, r.text


def test_x_api_token_header_also_accepted():
    client = _client(api_token="sekrit")
    r = client.post(
        "/api/funds/transfer",
        headers={"X-API-Token": "sekrit"},
        json={"direction": "to_futures", "amount_usdt": 10.0},
    )
    assert r.status_code == 200, r.text


def test_read_only_state_stays_open_with_token_configured():
    client = _client(api_token="sekrit")
    r = client.get("/api/state")
    assert r.status_code == 200
    assert "account" in r.json()


def test_cors_default_is_same_origin_only():
    client = _client("")
    r = client.options(
        "/api/state",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert "access-control-allow-origin" not in r.headers, (
        "no RS_CORS_ORIGINS -> cross-origin reads must be refused"
    )


def test_cors_allowlist_honours_env_override():
    client = _client("", cors_origins="https://ok.example")
    r = client.options(
        "/api/state",
        headers={
            "Origin": "https://ok.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert r.headers.get("access-control-allow-origin") == "https://ok.example"
