"""Unit tests for the CALC tool (plugins/calc_tool.py).

CalculatorTool is pure (no assistant/config needed) and uses a safe AST
evaluator, so it is the ideal deterministic anchor for both unit tests here and
the e2e tier.
"""
import pytest

from plugins.calc_tool import CalculatorTool
from tool_plugins import ToolStatus

pytestmark = pytest.mark.unit


@pytest.fixture
def calc():
    # BaseTool tolerates assistant=None (config becomes None); CALC never uses it.
    return CalculatorTool(assistant=None)


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("2+2", 4),
        ("17*3", 51),
        ("100/4", 25),
        ("85*0.15", 12.75),
        ("2**8", 256),
        ("-5 + 10", 5),
        ("10 x 5", 50),   # 'x' is normalized to '*'
        ("17 × 3", 51),   # unicode multiply
        ("100 ÷ 4", 25),  # unicode divide
    ],
)
async def test_arithmetic(calc, expression, expected):
    result = await calc.execute({"expression": expression})
    assert result.status == ToolStatus.SUCCESS
    assert result.data["result"] == expected
    # The spoken message must carry the numeric answer for the caller.
    assert str(result.data["result_str"]) in result.message


async def test_integer_valued_float_is_normalized(calc):
    result = await calc.execute({"expression": "10/2"})
    assert result.status == ToolStatus.SUCCESS
    assert result.data["result"] == 5
    assert result.data["result_str"] == "5"  # not "5.0"


async def test_division_by_zero(calc):
    result = await calc.execute({"expression": "1/0"})
    assert result.status == ToolStatus.FAILED
    assert "divide by zero" in result.message.lower()


async def test_empty_expression(calc):
    result = await calc.execute({"expression": ""})
    assert result.status == ToolStatus.FAILED


async def test_missing_expression(calc):
    result = await calc.execute({})
    assert result.status == ToolStatus.FAILED


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hi')",  # function call -> rejected
        "open('/etc/passwd')",                  # name/call -> rejected
        "1 + foo",                              # bare name -> rejected
        "[1,2,3]",                              # list literal -> rejected
    ],
)
async def test_rejects_unsafe_expressions(calc, expression):
    result = await calc.execute({"expression": expression})
    assert result.status == ToolStatus.FAILED


async def test_rejects_huge_exponent(calc):
    # Guards against CPU/memory DoS via giant powers (calc_tool MAX_EXPONENT).
    result = await calc.execute({"expression": "9**99999"})
    assert result.status == ToolStatus.FAILED
    assert "too large" in result.message.lower()
