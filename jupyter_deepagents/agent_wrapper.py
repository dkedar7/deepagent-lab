"""
Wrapper for LangGraph agent to provide a consistent API for the extension.
"""
import importlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from dotenv import load_dotenv
load_dotenv()


class AgentWrapper:
    """Wrapper class for LangGraph agent."""

    def __init__(self, agent_module_path: str = "my_agent", agent_variable_name: Optional[str] = None):
        """
        Initialize the agent wrapper.

        Args:
            agent_module_path: Path to the module containing the agent.
                              Defaults to "my_agent" which will import from my_agent.py
                              Can be overridden by JUPYTER_AGENT_PATH environment variable.
            agent_variable_name: Name of the variable to load from the module.
                                Defaults to None (will try 'agent' then 'graph').
                                Can be overridden by JUPYTER_AGENT_PATH environment variable.
        """
        self.agent = None

        # Check for environment variable: JUPYTER_AGENT_PATH="module_path:variable_name"
        env_path = os.environ.get('JUPYTER_AGENT_PATH')
        if env_path:
            parts = env_path.split(':', 1)
            if len(parts) == 2:
                self.agent_module_path = parts[0]
                self.agent_variable_name = parts[1]
                print(f"Using agent from environment: {self.agent_module_path}:{self.agent_variable_name}")
            else:
                print(f"Warning: JUPYTER_AGENT_PATH format should be 'module:variable', got: {env_path}")
                print(f"Using default: {agent_module_path}")
                self.agent_module_path = agent_module_path
                self.agent_variable_name = agent_variable_name
        else:
            self.agent_module_path = agent_module_path
            self.agent_variable_name = agent_variable_name

        self._load_agent()

    def _load_agent(self):
        """Load the agent from the specified module."""
        try:
            # Try to import the agent module
            module = importlib.import_module(self.agent_module_path)

            # If a specific variable name is provided, use it
            if self.agent_variable_name:
                if hasattr(module, self.agent_variable_name):
                    self.agent = getattr(module, self.agent_variable_name)
                    print(f"Loaded agent from {self.agent_module_path}.{self.agent_variable_name}")
                else:
                    raise AttributeError(
                        f"Module {self.agent_module_path} does not have '{self.agent_variable_name}' attribute"
                    )
            else:
                # Try default names: 'agent' then 'graph'
                if hasattr(module, 'agent'):
                    self.agent = module.agent
                    print(f"Loaded agent from {self.agent_module_path}.agent")
                elif hasattr(module, 'graph'):
                    self.agent = module.graph
                    print(f"Loaded agent from {self.agent_module_path}.graph")
                else:
                    raise AttributeError(
                        f"Module {self.agent_module_path} does not have 'agent' or 'graph' attribute"
                    )

        except ImportError as e:
            print(f"Warning: Could not import agent module '{self.agent_module_path}': {e}")
            print("Agent functionality will not be available until the module is created.")
            if os.environ.get('JUPYTER_AGENT_PATH'):
                print(f"Note: JUPYTER_AGENT_PATH is set to: {os.environ.get('JUPYTER_AGENT_PATH')}")
            self.agent = None
        except Exception as e:
            print(f"Error loading agent: {e}")
            self.agent = None

    def reload_agent(self):
        """Reload the agent module (useful for development)."""
        if self.agent_module_path in sys.modules:
            importlib.reload(sys.modules[self.agent_module_path])
        self._load_agent()

    def invoke(self, message: str, config: Optional[Dict[str, Any]] = None, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Invoke the agent with a message.

        Args:
            message: The user message to send to the agent
            config: Optional configuration for the agent
            thread_id: Optional thread ID for conversation history

        Returns:
            Dict containing the agent's response
        """
        if self.agent is None:
            error_msg = "Agent not loaded. "
            if os.environ.get('JUPYTER_AGENT_PATH'):
                error_msg += f"Check JUPYTER_AGENT_PATH: {os.environ.get('JUPYTER_AGENT_PATH')}"
            else:
                error_msg += "Please create my_agent.py with your LangGraph agent or set JUPYTER_AGENT_PATH."
            return {
                "error": error_msg,
                "status": "error"
            }

        try:
            # Prepare the input for the agent
            # Adjust this based on your agent's expected input format
            agent_input = {"messages": [{"role": "user", "content": message}]}

            # Prepare config with thread_id if provided
            agent_config = config or {}
            if thread_id:
                agent_config["configurable"] = agent_config.get("configurable", {})
                agent_config["configurable"]["thread_id"] = thread_id

            # Invoke the agent
            result = self.agent.invoke(agent_input, config=agent_config)

            # Extract the response
            # Adjust this based on your agent's output format
            response_text = ""

            if isinstance(result, dict):
                if "messages" in result and len(result["messages"]) > 0:
                    last_message = result["messages"][-1]

                    # Handle LangChain message objects (AIMessage, HumanMessage, etc.)
                    if hasattr(last_message, 'content'):
                        content = last_message.content
                        # Convert content to string if it's not already
                        if isinstance(content, str):
                            response_text = content
                        elif isinstance(content, list):
                            # Handle list of content blocks
                            response_text = " ".join(
                                block.get("text", str(block)) if isinstance(block, dict) else str(block)
                                for block in content
                            )
                        else:
                            response_text = str(content)
                    elif isinstance(last_message, dict):
                        response_text = last_message.get("content", str(last_message))
                    else:
                        response_text = str(last_message)
                else:
                    response_text = str(result)
            else:
                response_text = str(result)

            return {
                "response": response_text,
                "status": "success"
            }

        except Exception as e:
            return {
                "error": f"Error invoking agent: {str(e)}",
                "status": "error"
            }

    def stream(self, message: str, config: Optional[Dict[str, Any]] = None, thread_id: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        """
        Stream responses from the agent.

        Args:
            message: The user message to send to the agent
            config: Optional configuration for the agent
            thread_id: Optional thread ID for conversation history

        Yields:
            Dict containing chunks of the agent's response
        """
        if self.agent is None:
            error_msg = "Agent not loaded. "
            if os.environ.get('JUPYTER_AGENT_PATH'):
                error_msg += f"Check JUPYTER_AGENT_PATH: {os.environ.get('JUPYTER_AGENT_PATH')}"
            else:
                error_msg += "Please create my_agent.py with your LangGraph agent or set JUPYTER_AGENT_PATH."
            yield {
                "error": error_msg,
                "status": "error"
            }
            return

        try:
            # Prepare the input for the agent
            agent_input = {"messages": [{"role": "user", "content": message}]}

            # Prepare config with thread_id if provided
            agent_config = config or {}
            if thread_id:
                agent_config["configurable"] = agent_config.get("configurable", {})
                agent_config["configurable"]["thread_id"] = thread_id

            # Stream from the agent using "updates" mode to get intermediate steps
            for update in self.agent.stream(agent_input, config=agent_config, stream_mode="updates"):
                # update is a dict like {node_name: state_data}
                if isinstance(update, dict):
                    for node_name, state_data in update.items():
                        # Extract message content from the state update
                        if isinstance(state_data, dict) and "messages" in state_data:
                            messages = state_data["messages"]
                            if messages:
                                # Get the last message in this update
                                last_message = messages[-1] if isinstance(messages, list) else messages

                                # Check if this is a ToolMessage
                                message_type = last_message.__class__.__name__ if hasattr(last_message, '__class__') else None

                                # Handle ToolMessage (tool outputs)
                                # Skip ToolMessage entirely - don't send to frontend
                                if message_type == 'ToolMessage':
                                    # Don't yield anything for ToolMessages
                                    # They will not appear in the UI at all
                                    pass

                                # Handle regular messages (including AIMessage with tool calls)
                                elif hasattr(last_message, 'content'):
                                    content = last_message.content

                                    # Convert content to string if it's not already
                                    if isinstance(content, str):
                                        content_str = content
                                    elif isinstance(content, list):
                                        # Handle list of content blocks (e.g., [{"text": "...", "type": "text"}])
                                        content_str = " ".join(
                                            block.get("text", str(block)) if isinstance(block, dict) else str(block)
                                            for block in content
                                        )
                                    else:
                                        content_str = str(content)

                                    # Check for tool calls in AIMessage
                                    tool_calls = None
                                    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                                        tool_calls = []
                                        for tc in last_message.tool_calls:
                                            tool_calls.append({
                                                "id": tc.get("id") if isinstance(tc, dict) else getattr(tc, 'id', None),
                                                "name": tc.get("name") if isinstance(tc, dict) else getattr(tc, 'name', None),
                                                "args": tc.get("args") if isinstance(tc, dict) else getattr(tc, 'args', {})
                                            })

                                    # Clean content_str: strip whitespace
                                    content_str = content_str.strip() if content_str else ""

                                    # Filter out tool call dictionaries from content
                                    # These often appear as strings like "{'id': '...', 'input': {...}, 'name': '...', 'type': 'tool_use'}"
                                    if content_str and tool_calls:
                                        # Remove lines that look like tool call dictionaries
                                        import re
                                        # Pattern to match tool call dictionary representations
                                        tool_dict_pattern = r"\{'id':\s*'[^']+',\s*'input':\s*\{.*?\},\s*'name':\s*'[^']+',\s*'type':\s*'tool_use'\}"
                                        content_str = re.sub(tool_dict_pattern, '', content_str, flags=re.DOTALL)
                                        content_str = content_str.strip()

                                    # Yield tool calls (if any)
                                    if tool_calls:
                                        yield {
                                            "tool_calls": tool_calls,
                                            "node": node_name,
                                            "status": "streaming"
                                        }

                                    # Yield content separately, only if non-empty
                                    # This ensures tool call text doesn't appear in message content
                                    if content_str:
                                        yield {
                                            "chunk": content_str,
                                            "node": node_name,
                                            "status": "streaming"
                                        }

            yield {
                "status": "complete"
            }

        except Exception as e:
            yield {
                "error": f"Error streaming from agent: {str(e)}",
                "status": "error"
            }


# Global agent instance
_agent_instance: Optional[AgentWrapper] = None


def get_agent() -> AgentWrapper:
    """Get or create the global agent instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = AgentWrapper()
    return _agent_instance
