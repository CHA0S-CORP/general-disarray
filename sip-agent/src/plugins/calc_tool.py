"""
Calculator Tool Plugin
======================
Performs basic math calculations.

Usage in conversation:
User: "What's 15 percent of 85?"
LLM: [TOOL:CALC:expression=85*0.15]

User: "Calculate 123 plus 456"
LLM: [TOOL:CALC:expression=123+456]
"""

import ast
import math
import operator
from decimal import Decimal
from typing import Any, Dict, Tuple

from tool_plugins import BaseTool, ToolResult, ToolStatus


class CalculatorTool(BaseTool):
    """Perform mathematical calculations."""
    
    name = "CALC"
    description = "Calculate mathematical expressions (add, subtract, multiply, divide, percentages)"
    enabled = True
    speak_result = True  # informational: message is spoken in marker mode
    
    parameters = {
        "expression": {
            "type": "string",
            "description": "Math expression to evaluate (e.g., '15*0.15', '100/4', '2**8')",
            "required": True
        }
    }
    
    # Bounds for exponentiation to prevent CPU/memory DoS via huge powers.
    MAX_EXPONENT = 1000
    MAX_POW_BASE = 1_000_000
    # No intermediate or final result may exceed ~this many decimal digits
    # (checked BEFORE a power is computed, and on every integer result).
    MAX_RESULT_DIGITS = 400
    # Results at or above this magnitude are spoken in scientific notation.
    SCIENTIFIC_THRESHOLD = 1e15

    # Allowed operators for safe evaluation
    ALLOWED_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    def _safe_eval(self, node):
        """Safely evaluate an AST node."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return self._checked(node.value)
            raise ValueError(f"Invalid constant: {node.value}")

        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in self.ALLOWED_OPERATORS:
                raise ValueError(f"Operator not allowed: {op_type.__name__}")
            left = self._safe_eval(node.left)
            right = self._safe_eval(node.right)
            # Bound exponentiation: huge exponents (e.g. "9**9**9**9") produce
            # multi-million-digit integers that pin the CPU and balloon memory,
            # blocking the event loop. Reject anything that could be abusive.
            if op_type is ast.Pow:
                if abs(right) > self.MAX_EXPONENT or abs(left) > self.MAX_POW_BASE:
                    raise ValueError("Exponent or base too large")
                # Estimate the result size before computing it: 999**1000 is
                # ~3000 digits — useless to read out, and larger ones hit
                # Python's int->str digit limit or overflow floats.
                if abs(left) > 1 and right > 0 and \
                        right * math.log10(abs(left)) > self.MAX_RESULT_DIGITS:
                    raise OverflowError("result too large")
            return self._checked(self.ALLOWED_OPERATORS[op_type](left, right))
            
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in self.ALLOWED_OPERATORS:
                raise ValueError(f"Operator not allowed: {op_type.__name__}")
            operand = self._safe_eval(node.operand)
            return self._checked(self.ALLOWED_OPERATORS[op_type](operand))
            
        elif isinstance(node, ast.Expression):
            return self._safe_eval(node.body)
            
        else:
            raise ValueError(f"Invalid expression node: {type(node).__name__}")

    def _checked(self, value):
        """Reject results that can't be spoken: complex (e.g. (-8)**0.5),
        non-finite floats (inf/nan from float overflow), oversized ints."""
        if isinstance(value, complex):
            raise ValueError("result is not a real number")
        if isinstance(value, float) and not math.isfinite(value):
            raise OverflowError("result too large")
        if isinstance(value, int) and \
                value.bit_length() > self.MAX_RESULT_DIGITS * 3.33:
            raise OverflowError("result too large")
        return value

    @classmethod
    def _format_result(cls, result) -> Tuple[str, str]:
        """(result_str, spoken) for a finite numeric result.

        Integral floats collapse to ints; magnitudes >= 1e15 use scientific
        notation ("1.23457e+20", spoken "1.23457 times ten to the power of
        20") instead of reading out a 20+ digit number.
        """
        if isinstance(result, float) and result.is_integer() \
                and abs(result) < cls.SCIENTIFIC_THRESHOLD:
            result = int(result)
        if abs(result) >= cls.SCIENTIFIC_THRESHOLD:
            mantissa, _, exponent = f"{Decimal(result):.6e}".partition("e")
            mantissa = mantissa.rstrip("0").rstrip(".")
            exponent = int(exponent)
            exact = Decimal(f"{mantissa}e{exponent}") == Decimal(result)
            return (f"{mantissa}e{exponent:+d}",
                    f"{'' if exact else 'about '}{mantissa} times ten "
                    f"to the power of {exponent}")
        if isinstance(result, float):
            # 12 significant digits keeps real precision (123456.789,
            # 1000000.5) while rounding away binary-float noise so TTS
            # doesn't read "0.30000000000000004" aloud for 0.1+0.2.
            text = f"{result:.12g}"
            return text, text
        return str(result), str(result)
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        expression = params.get("expression", "")
        
        if not expression:
            return ToolResult(
                status=ToolStatus.FAILED,
                message="Please provide a math expression to calculate"
            )
            
        # Clean up the expression
        expression = expression.strip()
        
        # Replace common variations
        expression = expression.replace("×", "*").replace("÷", "/")
        expression = expression.replace("x", "*").replace("X", "*")
        
        try:
            # Parse the expression safely
            tree = ast.parse(expression, mode='eval')
            result = self._safe_eval(tree)
            
            result_str, spoken = self._format_result(result)
            if isinstance(result, float) and result.is_integer() \
                    and abs(result) < self.SCIENTIFIC_THRESHOLD:
                result = int(result)

            # Create a natural language response
            message = f"The result of {expression} is {spoken}"
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                message=message,
                data={
                    "expression": expression,
                    # Huge results are carried as their scientific string
                    # (a 400-digit int is JSON/log noise; float() of it is inf).
                    "result": result if abs(result) < self.SCIENTIFIC_THRESHOLD
                    else result_str,
                    "result_str": result_str
                }
            )
            
        except OverflowError:
            return ToolResult(
                status=ToolStatus.FAILED,
                message="That number is too large for me to work out."
            )
        except ZeroDivisionError:
            return ToolResult(
                status=ToolStatus.FAILED,
                message="Cannot divide by zero"
            )
        except (ValueError, SyntaxError) as e:
            return ToolResult(
                status=ToolStatus.FAILED,
                message=f"Invalid expression: {str(e)}"
            )
        except Exception as e:
            return ToolResult(
                status=ToolStatus.FAILED,
                message=f"Calculation error: {str(e)}"
            )
