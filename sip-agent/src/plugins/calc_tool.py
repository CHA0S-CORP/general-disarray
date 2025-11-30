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
import operator
from typing import Any, Dict

from tool_plugins import BaseTool, ToolResult, ToolStatus


class CalculatorTool(BaseTool):
    """Perform mathematical calculations."""
    
    name = "CALC"
    description = "Calculate mathematical expressions (add, subtract, multiply, divide, percentages)"
    enabled = True
    
    parameters = {
        "expression": {
            "type": "string",
            "description": "Math expression to evaluate (e.g., '15*0.15', '100/4', '2**8')",
            "required": True
        }
    }
    
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
        if isinstance(node, ast.Constant):  # Python 3.8+
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Invalid constant: {node.value}")
            
        elif isinstance(node, ast.Num):  # Python 3.7 compatibility
            return node.n
            
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in self.ALLOWED_OPERATORS:
                raise ValueError(f"Operator not allowed: {op_type.__name__}")
            left = self._safe_eval(node.left)
            right = self._safe_eval(node.right)
            return self.ALLOWED_OPERATORS[op_type](left, right)
            
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in self.ALLOWED_OPERATORS:
                raise ValueError(f"Operator not allowed: {op_type.__name__}")
            operand = self._safe_eval(node.operand)
            return self.ALLOWED_OPERATORS[op_type](operand)
            
        elif isinstance(node, ast.Expression):
            return self._safe_eval(node.body)
            
        else:
            raise ValueError(f"Invalid expression node: {type(node).__name__}")
    
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
            
            # Format the result nicely
            if isinstance(result, float):
                if result == int(result):
                    result = int(result)
                    result_str = str(result)
                else:
                    # Round to reasonable precision
                    result_str = f"{result:.6g}"
            else:
                result_str = str(result)
                
            # Create a natural language response
            message = f"The result of {expression} is {result_str}"
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                message=message,
                data={
                    "expression": expression,
                    "result": result,
                    "result_str": result_str
                }
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
