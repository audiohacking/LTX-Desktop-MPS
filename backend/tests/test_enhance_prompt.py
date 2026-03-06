"""Tests for prompt enhancement route."""
from __future__ import annotations

from tests.fakes.services import FakeResponse


def test_enhance_prompt_returns_enhanced_text(client, test_state, fake_services):
    """Enhance prompt route should return an enhanced version of the input."""
    test_state.state.app_settings.gemini_api_key = "test-gemini-key"

    fake_services.http.queue("post", FakeResponse(
        status_code=200,
        json_payload={
            "candidates": [{"content": {"parts": [{"text": "A cinematic shot of a majestic cat walking gracefully across a sun-drenched room, golden hour lighting, shallow depth of field"}]}}]
        },
    ))

    resp = client.post("/api/enhance-prompt", json={
        "prompt": "cat walking in room",
        "mode": "text-to-video",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "enhancedPrompt" in data
    assert len(data["enhancedPrompt"]) > len("cat walking in room")


def test_enhance_prompt_requires_gemini_key(client):
    """Should return error when Gemini key is not configured."""
    resp = client.post("/api/enhance-prompt", json={
        "prompt": "cat walking",
        "mode": "text-to-video",
    })
    assert resp.status_code == 400


def test_enhance_prompt_image_mode(client, test_state, fake_services):
    """Should work with image mode."""
    test_state.state.app_settings.gemini_api_key = "test-key"

    fake_services.http.queue("post", FakeResponse(
        status_code=200,
        json_payload={
            "candidates": [{"content": {"parts": [{"text": "A stunning photograph of a cat"}]}}]
        },
    ))

    resp = client.post("/api/enhance-prompt", json={
        "prompt": "cat photo",
        "mode": "text-to-image",
    })
    assert resp.status_code == 200
    assert "enhancedPrompt" in resp.json()
