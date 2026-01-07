import json
import logging
from dataclasses import dataclass
from typing import Any

from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.managers.io_struct import Function, Tool

def parse_tools(response: str, tools: list[dict[str, Any]], parser: str = "qwen"):
    """
    Parse tools from response and normalize to tau2-bench format.

    Returns calls with 'arguments' field (dict) instead of 'parameters'.
    """
    tools_list = [
        Tool(
            function=Function(
                name=tool["function"]["name"],
                description=tool["function"]["description"],
                parameters=tool["function"]["parameters"],
            ),
            type=tool["type"],
        )
        for tool in tools
    ]
    parser = FunctionCallParser(tools=tools_list, tool_call_parser=parser)
    normal_text, calls = parser.parse_non_stream(response)

    # Normalize calls: convert 'parameters' to 'arguments' and ensure it's a dict
    normalized_calls = []
    for call in calls:
        call_dict = call.model_dump()

        # Get parameters/arguments (might be dict or JSON string)
        params = call_dict.get("parameters", call_dict.get("arguments", {}))

        # Ensure it's a dict, not a JSON string
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Failed to parse parameters as JSON: {params}")
                params = {}

        # Create normalized call with 'arguments' field
        normalized_call = {
            "name": call_dict.get("name", ""),
            "arguments": params,  # tau2-bench expects 'arguments' as dict
            "id": call_dict.get("id", ""),
        }
        normalized_calls.append(normalized_call)

    return {
        "normal_text": normal_text,
        "calls": normalized_calls,
    }

logger = logging.getLogger(__name__)

@dataclass
class OpenAIToolCall:
    """OpenAI format tool call structure"""

    id: str
    type: str = "function"
    function: dict[str, Any] = None


@dataclass
class OpenAIAssistantMessage:
    """OpenAI format assistant message structure"""

    role: str = "assistant"
    content: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None


class OpenAICompatibleToolCallAdapter:
    """
    Adapter class that converts sglang tool call parsing results to OpenAI compatible format

    This class encapsulates existing tool call parsing and action conversion logic,
    and provides OpenAI format output interface.
    """

    def __init__(self, parser_type: str = "qwen"):
        """
        Initialize adapter

        Args:
            parser_type: Parser type, defaults to "qwen"
        """
        self.parser_type = parser_type

    def parse_response_to_openai_format(self, response: str, tools_schema: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Parse sglang response to OpenAI compatible format

        Args:
            response: Raw response text from sglang

        Returns:
            Dictionary containing OpenAI format message and parsing results

        Raises:
            Exception: Thrown when parsing fails
        """
        try:
            # Use existing parser to parse tool calls
            parsed = parse_tools(response, tools_schema, self.parser_type)

            # Extract parsing results
            normal_text = parsed["normal_text"]
            calls = parsed["calls"]

            # Convert to OpenAI format
            openai_message = self._convert_to_openai_message(normal_text, calls)

            return {"openai_message": openai_message, "parsed_result": parsed, "success": True}

        except Exception as e:
            logger.warning(f"Parsing failed with error: {str(e)}")
            return {"openai_message": None, "parsed_result": None, "success": False, "error": str(e)}

    def _convert_to_openai_message(self, normal_text: str, calls: list[dict[str, Any]]) -> OpenAIAssistantMessage:
        """
        Convert parsing results to OpenAI format assistant message

        Args:
            normal_text: Normal text content
            calls: List of tool calls

        Returns:
            OpenAI format assistant message
        """
        if not calls:
            # No tool calls, return plain text response
            return OpenAIAssistantMessage(role="assistant", content=normal_text, tool_calls=None)

        # Convert tool calls to OpenAI format
        openai_tool_calls = []
        for i, call in enumerate(calls):
            openai_tool_call = OpenAIToolCall(
                id=f"call_{i}_{call.get('name', 'unknown')}",
                type="function",
                function={"name": call.get("name", ""), "arguments": call.get("parameters", "{}")},
            )
            openai_tool_calls.append(openai_tool_call)

        result = OpenAIAssistantMessage(
            role="assistant", content=normal_text if normal_text.strip() else None, tool_calls=openai_tool_calls
        )
        return result

    def get_openai_tools_format(self, tools_schema: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Get OpenAI format tool definitions

        Returns:
            List of OpenAI format tools
        """
        openai_tools = []
        for tool in tools_schema:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool["function"]["name"],
                    "description": tool["function"]["description"],
                    "parameters": tool["function"]["parameters"],
                },
            }
            openai_tools.append(openai_tool)

        return openai_tools


# Usage examples and factory functions
def create_openai_adapter(
    parser_type: str = "qwen"
) -> OpenAICompatibleToolCallAdapter:
    """
    Factory function to create OpenAI compatible tool call adapter

    Args:
        parser_type: Parser type

    Returns:
        Configured adapter instance
    """
    return OpenAICompatibleToolCallAdapter(parser_type)
