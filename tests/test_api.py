import json

import ollama
import pytest
from fastapi.testclient import TestClient

from app import main


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "API_KEY", None)
    return TestClient(main.app)


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_calls_ollama_with_validated_input(client, monkeypatch):
    async def fake_chat(self, **kwargs):
        assert kwargs["model"] == "llama3.2"
        assert kwargs["messages"] == [{"role": "user", "content": "Hello"}]
        return ollama.ChatResponse(
            model="llama3.2",
            message=ollama.Message(role="assistant", content="Hi!"),
            done=True,
        )

    monkeypatch.setattr(ollama.AsyncClient, "chat", fake_chat)
    response = client.post(
        "/api/chat",
        json={"model": "llama3.2", "messages": [{"role": "user", "content": "Hello"}]},
    )

    assert response.status_code == 200
    assert response.json() == {
        "model": "llama3.2",
        "message": {"role": "assistant", "content": "Hi!"},
        "done": True,
    }


def test_chat_rejects_unsupported_model_name(client):
    response = client.post(
        "/api/chat",
        json={
            "model": "http://127.0.0.1/evil",
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )

    assert response.status_code == 422


def test_chat_rejects_oversized_request_body(client):
    payload = json.dumps(
        {"model": "llama3.2", "messages": [{"role": "user", "content": "x" * 17_000}]}
    )
    response = client.post(
        "/api/chat",
        content=payload,
        headers={"content-length": "300000"},
    )

    assert response.status_code == 413


def test_api_key_protects_api_routes(client, monkeypatch):
    monkeypatch.setattr(main, "API_KEY", "secret-value")

    assert client.get("/api/models").status_code == 401
    assert (
        client.get(
            "/api/models", headers={"Authorization": "Bearer secret-value"}
        ).status_code
        != 401
    )
    assert client.get("/health").status_code == 200
