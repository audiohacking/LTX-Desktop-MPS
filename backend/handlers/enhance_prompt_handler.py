"""Prompt enhancement handler (Gemini-powered)."""
from __future__ import annotations

import logging
from threading import RLock

from _routes._errors import HTTPError
from handlers.base import StateHandlerBase
from pydantic import BaseModel, Field, ValidationError
from services.interfaces import HTTPClient, HttpTimeoutError, JSONValue
from state.app_state_types import AppState

logger = logging.getLogger(__name__)


class _GeminiPart(BaseModel):
    text: str


class _GeminiContent(BaseModel):
    parts: list[_GeminiPart] = Field(min_length=1)


class _GeminiCandidate(BaseModel):
    content: _GeminiContent


class _GeminiResponsePayload(BaseModel):
    candidates: list[_GeminiCandidate] = Field(min_length=1)


def _extract_gemini_text(payload: object) -> str:
    try:
        parsed = _GeminiResponsePayload.model_validate(payload)
    except ValidationError:
        raise HTTPError(500, "GEMINI_PARSE_ERROR")
    return parsed.candidates[0].content.parts[0].text


class EnhancePromptHandler(StateHandlerBase):
    def __init__(self, state: AppState, lock: RLock, http: HTTPClient) -> None:
        super().__init__(state, lock)
        self._http = http

    def enhance(self, prompt: str, mode: str) -> dict[str, str]:
        gemini_api_key = self.state.app_settings.gemini_api_key
        if not gemini_api_key:
            raise HTTPError(400, "GEMINI_API_KEY_MISSING")

        is_image = mode in ("text-to-image", "t2i")

        system_text = (
            "You are a creative director's assistant. The user provides a rough prompt for "
            f"{'image' if is_image else 'video'} generation. Your job is to enhance it into a "
            "detailed, cinematic description that will produce stunning visual results.\n\n"
            "Guidelines:\n"
            "- Expand vague descriptions into specific, vivid details\n"
            "- Add lighting, camera angle, mood, and atmosphere\n"
            "- Keep the core intent of the original prompt\n"
            "- Write 2-4 sentences max\n"
            "- Write only the enhanced prompt, no labels or explanations\n"
        )

        gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        gemini_payload: dict[str, JSONValue] = {
            "contents": [{"role": "user", "parts": [{"text": f"Enhance this prompt: {prompt}"}]}],
            "systemInstruction": {"parts": [{"text": system_text}]},
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 256},
        }

        try:
            response = self._http.post(
                gemini_url,
                headers={"Content-Type": "application/json", "x-goog-api-key": gemini_api_key},
                json_payload=gemini_payload,
                timeout=30,
            )
        except HttpTimeoutError as exc:
            raise HTTPError(504, "Gemini API request timed out") from exc
        except Exception as exc:
            raise HTTPError(500, str(exc)) from exc

        if response.status_code != 200:
            raise HTTPError(response.status_code, f"Gemini API error: {response.text}")

        enhanced = _extract_gemini_text(response.json()).strip()
        return {"enhancedPrompt": enhanced}
