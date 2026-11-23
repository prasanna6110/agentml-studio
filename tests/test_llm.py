import json

import httpx

from agentml.llm import GeminiLLM, Message, MockLLM, OpenAILLM, ToolCall, extract_json

TOOLS = [{"name": "calculator", "description": "Evaluate math",
          "parameters": {"type": "object", "properties": {"expression": {"type": "string", "default": "1"}},
                         "required": ["expression"], "additionalProperties": False}}]


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure!\n```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert extract_json('text before {"x": "has } brace"} after') == {"x": "has } brace"}
    assert extract_json("no json here") is None


def test_mock_echoes_default_decision():
    out = MockLLM().complete('Decide.\n<default_decision>\n{"k": 3}\n</default_decision>')
    assert json.loads(out) == {"k": 3}


def test_mock_routes_to_tool_and_summarises():
    llm = MockLLM()
    r = llm.chat([Message.user("please calculate 2 + 3 * 4")], tools=TOOLS)
    assert r.tool_calls and r.tool_calls[0].name == "calculator"
    assert r.tool_calls[0].arguments["expression"] == "2 + 3 * 4"
    r2 = llm.chat([Message.user("x"), Message.assistant("", r.tool_calls),
                   Message.tool("14", r.tool_calls[0].id, "calculator")], tools=TOOLS)
    assert "14" in r2.content and not r2.tool_calls
    assert llm.total_usage["calls"] == 2


def _transport(check, reply):
    def handler(request: httpx.Request) -> httpx.Response:
        check(request, json.loads(request.content))
        return httpx.Response(200, json=reply)
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_wire_format_and_parsing():
    def check(req, body):
        assert req.headers["authorization"] == "Bearer k"
        assert body["tools"][0]["type"] == "function"
        assert body["messages"][1]["tool_calls"][0]["function"]["arguments"] == '{"expression": "1"}'

    reply = {"choices": [{"message": {"content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "calculator", "arguments": '{"expression": "2*3"}'}}]},
        "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}
    llm = OpenAILLM("k", "gpt-test", client=_transport(check, reply))
    r = llm.chat([Message.user("hi"), Message.assistant("", [ToolCall("calculator", {"expression": "1"}, "c0")]),
                  Message.tool("1", "c0", "calculator")], tools=TOOLS)
    assert r.tool_calls[0].arguments == {"expression": "2*3"} and r.content == ""


def test_gemini_strips_unsupported_schema_keys():
    def check(req, body):
        params = body["tools"][0]["functionDeclarations"][0]["parameters"]
        assert "additionalProperties" not in params
        assert "default" not in params["properties"]["expression"]
        assert body["systemInstruction"]["parts"][0]["text"] == "sys"

    reply = {"candidates": [{"content": {"parts": [{"text": "hello"}]}, "finishReason": "STOP"}],
             "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2}}
    llm = GeminiLLM("k", "gemini-test", client=_transport(check, reply))
    assert llm.chat([Message.system("sys"), Message.user("hi")], tools=TOOLS).content == "hello"


def test_openai_adapts_to_new_model_parameters():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if "max_tokens" in body:
            return httpx.Response(400, json={"error": {"message": "Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens'."}})
        if "temperature" in body:
            return httpx.Response(400, json={"error": {"message": "Unsupported value: 'temperature' does not support 0.2"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}], "usage": {}})

    llm = OpenAILLM("k", "gpt-new", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert llm.complete("hi") == "OK"
    assert "max_completion_tokens" in calls[-1] and "temperature" not in calls[-1]
    n = len(calls)
    assert llm.complete("again") == "OK" and len(calls) == n + 1  # remembered, no extra round-trips


def test_missing_model_name_is_a_clear_error():
    import pytest

    from agentml.config import Settings
    from agentml.llm import LLMError, get_llm

    with pytest.raises(LLMError, match="No model name"):
        get_llm("ollama", "", settings=Settings(provider="ollama", model=""))
