"""
LLM Engine
==========
Handles LLM inference with tool calling support.
Supports multiple backends: vLLM, Ollama, LM Studio.
"""

import re
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

try:
    from openai import AsyncOpenAI
    OPENAI_CLIENT_AVAILABLE = True
except ImportError:
    OPENAI_CLIENT_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from config import Config




logger = logging.getLogger(__name__)

@dataclass
class ToolCall:
    """Parsed tool call from LLM response."""
    name: str
    params: Dict[str, Any]
    raw: str


class LLMEngine:
    """LLM inference engine with tool support."""
    
    def __init__(self, config: Config, tool_manager: 'ToolManager'):
        self.config = config
        self.tool_manager = tool_manager
        self.client: Optional[AsyncOpenAI] = None
        
    async def start(self):
        """Initialize the LLM client."""
        if not OPENAI_CLIENT_AVAILABLE:
            logger.warning("OpenAI client not available, using mock LLM")
            return
            
        # Create OpenAI-compatible client for local LLM
        self.client = AsyncOpenAI(
            base_url=self.config.llm_base_url,
            api_key=self.config.llm_api_key,
            timeout=60.0
        )
        
        # Test connection
        try:
            models = await self.client.models.list()
            logger.info(f"Connected to LLM backend. Available models: {[m.id for m in models.data]}")
        except Exception as e:
            logger.warning(f"Could not connect to LLM backend: {e}")
            logger.info("Will retry on first request")
            
    async def stop(self):
        """Cleanup."""
        if self.client:
            await self.client.close()
            
    async def generate_greeting(self) -> str:
        """Generate a greeting for incoming calls."""
        greetings = [
            "Hello! This is your AI assistant. How can I help you today?",
            "Hi there! I'm your AI assistant. What can I do for you?",
            "Hello! I'm here to help. What would you like to talk about?",
        ]

        better_greeting = [
            "Whats up G! Tell the clanka what ya want!",
            "Resistance is futile! State your business!",
            "Ready to create chaos, Professor! What do you need?",
            "Standing by for commands"
        ]
        
        # Could use LLM for dynamic greeting, but static is faster
        import random
        return random.choice(better_greeting)
        
    async def generate_response(
        self,
        conversation_history: List[Dict[str, str]],
        call_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a response to the conversation."""
        
        # Build messages with system prompt
        messages = [
            {"role": "system", "content": self._build_system_prompt(call_context)}
        ]
        
        # Add conversation history (limited to max turns)
        messages.extend(
            conversation_history[-self.config.max_conversation_turns * 2:]
        )
        
        # Generate response
        response_text = await self._generate(messages)
        
        # Parse and execute any tool calls
        response_text, tool_results = await self._process_tool_calls(response_text)
        
        # If tools were called, we might want to incorporate results
        if tool_results:
            # For voice, we typically just confirm the action was taken
            # The tool call is already parsed out of the response
            pass
            
        return response_text
        
    def _build_system_prompt(self, call_context: Optional[Dict[str, Any]] = None) -> str:
        """Build system prompt with context."""
        prompt = self.config.system_prompt
        
        # Add time context
        now = datetime.now()
        prompt += f"\n\nCurrent time: {now.strftime('%I:%M %p on %A, %B %d, %Y')}"
        
        # Add call context
        if call_context:
            prompt += f"\n\nCall information:"
            prompt += f"\n- Caller: {call_context.get('remote_uri', 'unknown')}"
            prompt += f"\n- Duration: {call_context.get('duration', 0):.0f} seconds"
            
        # Add available tools
        prompt += "\n\nAvailable tools and their status:"
        for tool_name, tool in self.tool_manager.tools.items():
            status = "enabled" if tool.enabled else "disabled"
            prompt += f"\n- {tool_name}: {status}"
            
        return prompt
        
    async def _generate(self, messages: List[Dict[str, str]]) -> str:
        """Call the LLM to generate a response."""
        if not self.client:
            # Mock response
            return self._mock_response(messages)
            
        try:
            response = await self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=messages,
                max_tokens=self.config.llm_max_tokens,
                temperature=self.config.llm_temperature,
                top_p=self.config.llm_top_p
            )
            
            # --- CRITICAL FIX ---
            # gpt-oss-20b / vLLM can return None for content if it gets confused 
            # or tries to use native tools. We must fallback to empty string.
            content = response.choices[0].message.content
            
            # --- FIX: Handle Empty Content / Length Finish ---
            if content is None or not content.strip():
                finish_reason = response.choices[0].finish_reason
                logger.warning(f"LLM returned empty content. Reason: {finish_reason}")
                
                # If it ran out of tokens while thinking, we can't recover easily 
                # without more tokens, so we give a polite error.
                if finish_reason == 'length':
                    return "I'm sorry, I was thinking too hard and ran out of time. Could you ask that again?"
                
                return "I didn't catch that. Could you repeat it?"

            return content
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return "I'm sorry, I'm having trouble processing that. Could you try again?"
            
    def _mock_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate mock response when LLM unavailable."""
        last_user_msg = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                last_user_msg = msg["content"].lower()
                break
                
        # Simple keyword matching for testing
        if "timer" in last_user_msg or "remind" in last_user_msg:
            return "I'll set that timer for you. [TOOL:SET_TIMER:duration=300,message=Timer complete]"
        elif "call" in last_user_msg and "back" in last_user_msg:
            return "I'll call you back. [TOOL:CALLBACK:delay=60,message=Callback as requested]"
        elif "bye" in last_user_msg or "goodbye" in last_user_msg:
            return "Goodbye! Have a great day! [TOOL:HANGUP]"
        elif "help" in last_user_msg:
            return "I can help you with timers, reminders, and callbacks. What would you like me to do?"
        else:
            return "I understand. Is there anything specific I can help you with?"
            
    async def _process_tool_calls(self, response: str) -> Tuple[str, List[Dict]]:
        """Parse and execute tool calls from response."""
        tool_results = []
        
        # Find tool calls in format: [TOOL:name:param1=val1,param2=val2] or [TOOL:name]
        pattern_with_params = r'\[TOOL:(\w+):([^\]]+)\]'
        pattern_no_params = r'\[TOOL:(\w+)\]'
        
        # Process tools with parameters
        matches = list(re.finditer(pattern_with_params, response))
        for match in matches:
            tool_name = match.group(1)
            params_str = match.group(2)
            
            # Parse parameters
            params = {}
            for param in params_str.split(','):
                if '=' in param:
                    key, value = param.split('=', 1)
                    value = self._parse_param_value(value)
                    params[key.strip()] = value
                    
            tool_call = ToolCall(
                name=tool_name,
                params=params,
                raw=match.group(0)
            )
            
            try:
                result = await self.tool_manager.execute_tool(tool_call)
                tool_results.append({
                    "tool": tool_name,
                    "params": params,
                    "result": result
                })
            except Exception as e:
                logger.error(f"Tool execution error: {e}")
                tool_results.append({
                    "tool": tool_name,
                    "params": params,
                    "error": str(e)
                })
        
        # Process tools without parameters (e.g., HANGUP)
        # Remove already-matched sections first to avoid double-matching
        temp_response = re.sub(pattern_with_params, '', response)
        matches_no_params = list(re.finditer(pattern_no_params, temp_response))
        
        for match in matches_no_params:
            tool_name = match.group(1)
            
            tool_call = ToolCall(
                name=tool_name,
                params={},
                raw=match.group(0)
            )
            
            try:
                result = await self.tool_manager.execute_tool(tool_call)
                tool_results.append({
                    "tool": tool_name,
                    "params": {},
                    "result": result
                })
            except Exception as e:
                logger.error(f"Tool execution error: {e}")
                tool_results.append({
                    "tool": tool_name,
                    "params": {},
                    "error": str(e)
                })
                
        # Remove all tool calls from response text
        clean_response = re.sub(pattern_with_params, '', response)
        clean_response = re.sub(pattern_no_params, '', clean_response).strip()
        
        return clean_response, tool_results
        
    def _parse_param_value(self, value: str) -> Any:
        """Parse parameter value to appropriate type."""
        value = value.strip()
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
            
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
            
        # Try boolean
        if value.lower() in ('true', 'yes'):
            return True
        if value.lower() in ('false', 'no'):
            return False
            
        # Return as string
        return value


class OllamaEngine(LLMEngine):
    """
    Alternative engine using Ollama directly.
    Useful if not running vLLM.
    """
    
    def __init__(self, config: Config, tool_manager: 'ToolManager'):
        super().__init__(config, tool_manager)
        self.ollama_url = config.llm_base_url.replace('/v1', '')
        
    async def _generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate using Ollama API."""
        if not HTTPX_AVAILABLE:
            return self._mock_response(messages)
            
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ollama_url}/api/chat",
                    json={
                        "model": self.config.llm_model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "num_predict": self.config.llm_max_tokens,
                            "temperature": self.config.llm_temperature,
                            "top_p": self.config.llm_top_p
                        }
                    },
                    timeout=60.0
                )
                response.raise_for_status()
                data = response.json()
                return data["message"]["content"]
                
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return "I'm sorry, I'm having trouble processing that."


class LMStudioEngine(LLMEngine):
    """
    Alternative engine for LM Studio.
    LM Studio provides OpenAI-compatible API on port 1234 by default.
    """
    
    def __init__(self, config: Config, tool_manager: 'ToolManager'):
        config.llm_base_url = config.llm_base_url or "http://localhost:1234/v1"
        super().__init__(config, tool_manager)


# Factory function
def create_llm_engine(config: Config, tool_manager: 'ToolManager') -> LLMEngine:
    """Create appropriate LLM engine based on config."""
    backend = config.llm_backend.lower()
    
    if backend == "ollama":
        return OllamaEngine(config, tool_manager)
    elif backend == "lmstudio":
        return LMStudioEngine(config, tool_manager)
    else:  # vllm or default
        return LLMEngine(config, tool_manager)
