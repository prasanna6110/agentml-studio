import pytest

from agentml.tools import ToolRegistry, calculator, tool


@tool
def greet(name: str, times: int = 1, loud: bool | None = None) -> str:
    """Greet someone.

    Args:
        name: Who to greet.
        times: Repetitions.
    """
    return " ".join([f"hi {name}"] * times)


def test_schema_from_signature():
    s = greet.tool.schema
    assert s["name"] == "greet" and s["description"] == "Greet someone."
    props = s["parameters"]["properties"]
    assert props["name"] == {"type": "string", "description": "Who to greet."}
    assert props["times"]["type"] == "integer" and props["times"]["default"] == 1
    assert props["loud"]["type"] == "boolean"
    assert s["parameters"]["required"] == ["name"]


def test_registry_execute_handles_errors_and_unknown_args():
    reg = ToolRegistry([greet, calculator])
    assert reg.execute("greet", {"name": "Prasanna", "times": 2, "junk": 1}).output == "hi Prasanna hi Prasanna"
    bad = reg.execute("calculator", {"expression": "1/0"})
    assert not bad.ok and "ZeroDivisionError" in bad.output
    assert not reg.execute("nope", {}).ok


@pytest.mark.parametrize("expr,val", [("2+3*4", 14), ("sqrt(16) + 2**3", 12.0), ("-(5 % 3)", -2), ("round(pi, 2)", 3.14)])
def test_calculator(expr, val):
    assert calculator(expr) == pytest.approx(val)


@pytest.mark.parametrize("expr", ["__import__('os').system('ls')", "open('x')", "9**9999", "[1,2]"])
def test_calculator_rejects_unsafe(expr):
    with pytest.raises(ValueError):
        calculator(expr)
