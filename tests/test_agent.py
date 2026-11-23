from agentml.agents import Agent, LongTermMemory, ShortTermMemory
from agentml.llm import LLM, LLMResponse, Message, ToolCall
from agentml.tools import ToolRegistry, calculator


class ScriptedLLM(LLM):
    """Replays a fixed list of responses — lets us test multi-step ReAct behaviour exactly."""

    provider = "scripted"

    def __init__(self, responses):
        super().__init__("scripted")
        self.responses = list(responses)
        self.seen = []

    def _chat(self, messages, tools, temperature, max_tokens):
        self.seen.append(list(messages))
        return self.responses.pop(0)


def test_react_loop_multi_step():
    llm = ScriptedLLM([
        LLMResponse("Let me compute.", [ToolCall("calculator", {"expression": "6*7"}, "a")]),
        LLMResponse("", [ToolCall("calculator", {"expression": "42+1"}, "b")]),
        LLMResponse("The answer is 43."),
    ])
    agent = Agent("Tester", "a test agent", llm, tools=ToolRegistry([calculator]))
    res = agent.run("compute 6*7+1")
    assert res.answer == "The answer is 43." and res.tool_calls == 2
    # the second LLM call must see the first tool's result
    assert any(m.role == "tool" and m.content == "42" for m in llm.seen[1])
    kinds = [s.kind for s in res.steps]
    assert kinds == ["thought", "tool_call", "tool_result", "tool_call", "tool_result", "answer"]


def test_step_limit_forces_final_answer():
    loop = [LLMResponse("", [ToolCall("calculator", {"expression": "1"}, str(i))]) for i in range(3)]
    llm = ScriptedLLM(loop + [LLMResponse("best effort")])
    res = Agent("T", "t", llm, tools=ToolRegistry([calculator]), max_steps=3).run("loop forever")
    assert res.answer == "best effort" and res.tool_calls == 3


def test_decide_merges_and_validates():
    llm = ScriptedLLM([LLMResponse('```json\n{"metric": "accuracy", "evil": 1, "rationale": "ok"}\n```')])
    d = Agent("P", "planner", llm).decide("q", {"metric": "f1", "target": "y"}, {"e": 1})
    assert d == {"metric": "accuracy", "target": "y", "rationale": "ok"}
    bad = ScriptedLLM([LLMResponse("I refuse to answer in JSON")])
    assert Agent("P", "p", bad).decide("q", {"a": 1}, "ev") == {"a": 1}


def test_short_term_memory_never_starts_with_orphan_tool():
    m = ShortTermMemory(max_messages=3)
    m.add(Message.system("s"))
    m.add(Message.user("u"))
    m.add(Message.assistant("", [ToolCall("t", {}, "1")]))
    m.add(Message.tool("r", "1", "t"))
    m.add(Message.assistant("done"))
    m.add(Message.user("next"))
    assert m.messages[0].role == "system" and m.messages[1].role != "tool"


def test_long_term_memory_recall_and_persist(tmp_path):
    p = tmp_path / "mem.json"
    mem = LongTermMemory(p)
    mem.remember("gradient boosting won on the churn dataset", kind="experiment")
    mem.remember("the user prefers concise answers", kind="preference")
    top, score = mem.recall("which model won churn?", k=1)[0]
    assert "boosting" in top.text and score > 0
    assert len(LongTermMemory(p)) == 2
