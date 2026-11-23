import base64
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from agentml.imagegen import ImageGenerator, ImageGenError, clean_prompt, guess_size
from agentml.server.app import create_app

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 64).decode()


def client_for(responder):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        return responder(body)

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


def test_prompt_helpers():
    assert clean_prompt("Generate an image of a cat on a roof") == "a cat on a roof"
    assert clean_prompt("ek robot ki image banao") == "ek robot"
    assert clean_prompt("ek robot ki image banao jo solar panel saaf kar raha hai") == "ek robot jo solar panel saaf kar raha hai"
    assert guess_size("a youtube thumbnail of a robot") == "1536x1024"
    assert guess_size("a poster for tech fest") == "1024x1536"
    assert guess_size("a cat") == "1024x1024"


def test_generate_saves_png_and_gallery(tmp_path):
    http, calls = client_for(lambda b: httpx.Response(200, json={"data": [{"b64_json": PNG}]}))
    gen = ImageGenerator("sk-test", tmp_path, client=http)
    rec = gen.generate("Draw a logo for AgentML", quality="high")
    assert calls[0]["model"] == "gpt-image-1" and calls[0]["quality"] == "high"
    assert (tmp_path / rec["file"]).read_bytes().startswith(b"\x89PNG")
    assert gen.gallery()[0]["file"] == rec["file"] and rec["subject"] == "AgentML"


def test_falls_back_to_dalle_when_first_model_is_unavailable(tmp_path):
    def responder(body):
        if body["model"] == "gpt-image-1":
            return httpx.Response(403, json={"error": {"message": "Your organization must be verified"}})
        assert body["response_format"] == "b64_json"
        return httpx.Response(200, json={"data": [{"b64_json": PNG, "revised_prompt": "A neat logo"}]})

    http, calls = client_for(responder)
    rec = ImageGenerator("sk-test", tmp_path, client=http).generate("a logo")
    assert [c["model"] for c in calls] == ["gpt-image-1", "dall-e-3"] and rec["model"] == "dall-e-3"
    assert rec["revised_prompt"] == "A neat logo"


def test_bad_key_and_missing_key(tmp_path):
    http, calls = client_for(lambda b: httpx.Response(401, json={"error": {"message": "Incorrect API key provided"}}))
    with pytest.raises(ImageGenError, match="Incorrect API key"):
        ImageGenerator("sk-bad", tmp_path, client=http).generate("a cat")
    assert len(calls) == 1  # no pointless retry with the second model
    with pytest.raises(ImageGenError, match="needs an OpenAI API key"):
        ImageGenerator(None, tmp_path).generate("a cat")


def test_api_without_key_explains_what_to_do(settings, llm):
    settings.openai_api_key = None
    client = TestClient(create_app(settings, llm, seed_examples=False))
    r = client.post("/api/images", json={"prompt": "a robot"})
    assert r.status_code == 400 and "Settings" in r.json()["detail"]
    assert client.get("/api/images").json() == []
    assert client.get("/api/images/../../secret.png").status_code == 404


def test_real_llm_rewrites_the_prompt(tmp_path):
    from agentml.llm import LLM, LLMResponse

    class Writer(LLM):
        provider = "openai"

        def _chat(self, messages, tools, temperature, max_tokens):
            return LLMResponse("A friendly robot cleaning solar panels on a rooftop at golden hour, digital art")

    http, calls = client_for(lambda b: httpx.Response(200, json={"data": [{"b64_json": PNG}]}))
    rec = ImageGenerator("sk-test", tmp_path, client=http).generate("robot ki image banao", llm=Writer("w"))
    assert calls[0]["prompt"].startswith("A friendly robot") and rec["subject"] == calls[0]["prompt"]
