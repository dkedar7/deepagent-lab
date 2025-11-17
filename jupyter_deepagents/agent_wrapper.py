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

    def set_root_dir(self, root_dir: str):
        """
        Set the root directory on the agent's backend if it has one.

        Args:
            root_dir: The root directory path (JupyterLab launch directory)
        """
        if self.agent and hasattr(self.agent, 'backend'):
            try:
                # Import FilesystemBackend dynamically
                from deepagents.tools.filesystem import FilesystemBackend
                # Update the backend's root_dir
                self.agent.backend = FilesystemBackend(root_dir=root_dir, virtual_mode=True)
                print(f"Set agent backend root_dir to: {root_dir}")
            except ImportError:
                # FilesystemBackend not available, skip
                pass
            except Exception as e:
                print(f"Warning: Could not set agent backend root_dir: {e}")

    def _append_context_to_message(self, message: str, context: Optional[Dict[str, Any]]) -> str:
        """
        Append context information to the message.

        Args:
            message: The original user message
            context: Context dict with current_directory and focused_widget

        Returns:
            Message with appended context
        """
        if not context:
            return message

        context_parts = []
        if context.get("current_directory"):
            context_parts.append(f"Current directory: {context['current_directory']}")
        if context.get("focused_widget"):
            focused = context['focused_widget']
            # Check if it's a file path or special widget
            if '/' in focused or focused.endswith(('.ipynb', '.py', '.txt', '.md')):
                context_parts.append(f"Currently focused file: {focused}")
            else:
                context_parts.append(f"Currently focused: {focused}")

        if context_parts:
            return f"{message}\n\n" + "\n".join(context_parts)
        return message

    def invoke(self, message: str, config: Optional[Dict[str, Any]] = None, thread_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
            # Append context to message
            message_with_context = self._append_context_to_message(message, context)

            # Prepare the input for the agent
            # Adjust this based on your agent's expected input format
            agent_input = {"messages": [{"role": "user", "content": message_with_context}]}

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

    def resume_from_interrupt(self, decisions: list, config: Optional[Dict[str, Any]] = None, thread_id: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        """
        Resume execution after a human-in-the-loop interrupt.

        Args:
            decisions: List of decision objects with 'type' and optional fields
            config: Optional configuration for the agent
            thread_id: Thread ID to resume

        Yields:
            Dict containing chunks of the agent's response
        """
        if self.agent is None:
            yield {
                "error": "Agent not loaded",
                "status": "error"
            }
            return

        try:
            from langgraph.types import Command

            # Prepare config with thread_id
            agent_config = config or {}
            if thread_id:
                agent_config["configurable"] = agent_config.get("configurable", {})
                agent_config["configurable"]["thread_id"] = thread_id

            # Create resume command
            resume_input = Command(resume={"decisions": decisions})

            # Stream from the agent after resuming
            for update in self.agent.stream(resume_input, config=agent_config, stream_mode="updates"):
                # Check for interrupts again
                if isinstance(update, dict) and "__interrupt__" in update:
                    interrupt_value = update["__interrupt__"]

                    # Handle different formats (same as in stream method)
                    if isinstance(interrupt_value, tuple):
                        if len(interrupt_value) == 1:
                            # Single-element tuple containing Interrupt object
                            interrupt_obj = interrupt_value[0]
                            if hasattr(interrupt_obj, 'value') and isinstance(interrupt_obj.value, dict):
                                action_requests = interrupt_obj.value.get('action_requests', [])
                                review_configs = interrupt_obj.value.get('review_configs', [])
                            else:
                                action_requests = getattr(interrupt_obj, 'action_requests', [])
                                review_configs = getattr(interrupt_obj, 'review_configs', [])
                        elif len(interrupt_value) == 2:
                            # Two-element tuple: (action_requests, review_configs)
                            action_requests, review_configs = interrupt_value
                        else:
                            action_requests = []
                            review_configs = []
                    else:
                        # Handle object format
                        action_requests = getattr(interrupt_value, 'action_requests', [])
                        review_configs = getattr(interrupt_value, 'review_configs', [])

                    # Convert to dict for JSON serialization
                    interrupt_data = {
                        "action_requests": [],
                        "review_configs": []
                    }

                    # Extract action requests
                    for i, action in enumerate(action_requests):
                        # Handle both dict and object formats, and both 'name' and 'tool' field names
                        if isinstance(action, dict):
                            tool_name = action.get('tool') or action.get('name')
                            tool_call_id = action.get('tool_call_id', f"call_{i}")
                            args = action.get('args', {})
                            description = action.get('description')
                        else:
                            tool_name = getattr(action, 'tool', None) or getattr(action, 'name', None)
                            tool_call_id = getattr(action, 'tool_call_id', f"call_{i}")
                            args = getattr(action, 'args', {})
                            description = getattr(action, 'description', None)

                        interrupt_data["action_requests"].append({
                            "tool": tool_name,
                            "tool_call_id": tool_call_id,
                            "args": args,
                            "description": description
                        })

                    # Extract review configs
                    for config in review_configs:
                        interrupt_data["review_configs"].append({
                            "allowed_decisions": getattr(config, 'allowed_decisions', config.get('allowed_decisions') if isinstance(config, dict) else [])
                        })

                    yield {
                        "interrupt": interrupt_data,
                        "status": "interrupt"
                    }
                    continue

                # Regular update processing (same as stream method)
                if isinstance(update, dict):
                    for node_name, state_data in update.items():
                        if isinstance(state_data, dict) and "messages" in state_data:
                            messages = state_data["messages"]
                            if messages:
                                last_message = messages[-1] if isinstance(messages, list) else messages
                                message_type = last_message.__class__.__name__ if hasattr(last_message, '__class__') else None

                                if message_type == 'ToolMessage':
                                    pass
                                elif hasattr(last_message, 'content'):
                                    content = last_message.content
                                    if isinstance(content, str):
                                        content_str = content
                                    elif isinstance(content, list):
                                        content_str = " ".join(
                                            block.get("text", str(block)) if isinstance(block, dict) else str(block)
                                            for block in content
                                        )
                                    else:
                                        content_str = str(content)

                                    tool_calls = None
                                    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                                        tool_calls = []
                                        for tc in last_message.tool_calls:
                                            tool_calls.append({
                                                "id": tc.get("id") if isinstance(tc, dict) else getattr(tc, 'id', None),
                                                "name": tc.get("name") if isinstance(tc, dict) else getattr(tc, 'name', None),
                                                "args": tc.get("args") if isinstance(tc, dict) else getattr(tc, 'args', {})
                                            })

                                    content_str = content_str.strip() if content_str else ""

                                    if content_str and tool_calls:
                                        import re
                                        tool_dict_pattern = r"\{'id':\s*'[^']+',\s*'input':\s*\{.*?\},\s*'name':\s*'[^']+',\s*'type':\s*'tool_use'\}"
                                        content_str = re.sub(tool_dict_pattern, '', content_str, flags=re.DOTALL)
                                        content_str = content_str.strip()

                                    if tool_calls:
                                        yield {
                                            "tool_calls": tool_calls,
                                            "node": node_name,
                                            "status": "streaming"
                                        }

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
                "error": f"Error resuming from interrupt: {str(e)}",
                "status": "error"
            }

    def stream(self, message: str, config: Optional[Dict[str, Any]] = None, thread_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
        """
        Stream responses from the agent.

        Args:
            message: The user message to send to the agent
            config: Optional configuration for the agent
            thread_id: Optional thread ID for conversation history
            context: Optional context with current_directory and focused_notebook

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
            # Append context to message
            message_with_context = self._append_context_to_message(message, context)

            # Prepare the input for the agent
            agent_input = {"messages": [{"role": "user", "content": message_with_context}]}

            # Prepare config with thread_id if provided
            agent_config = config or {}
            if thread_id:
                agent_config["configurable"] = agent_config.get("configurable", {})
                agent_config["configurable"]["thread_id"] = thread_id

            # Stream from the agent using "updates" mode to get intermediate steps
            for update in self.agent.stream(agent_input, config=agent_config, stream_mode="updates"):
                # Check for interrupts (human-in-the-loop)
                if isinstance(update, dict) and "__interrupt__" in update:
                    interrupt_value = update["__interrupt__"]
                    print(f"DEBUG: Interrupt detected. Type: {type(interrupt_value)}, Value: {interrupt_value}")

                    # Handle different formats
                    if isinstance(interrupt_value, tuple):
                        if len(interrupt_value) == 1:
                            # Single-element tuple containing Interrupt object
                            interrupt_obj = interrupt_value[0]
                            if hasattr(interrupt_obj, 'value') and isinstance(interrupt_obj.value, dict):
                                action_requests = interrupt_obj.value.get('action_requests', [])
                                review_configs = interrupt_obj.value.get('review_configs', [])
                                print(f"DEBUG: 1-tuple format with Interrupt object - Actions: {len(action_requests)}, Configs: {len(review_configs)}")
                            else:
                                action_requests = getattr(interrupt_obj, 'action_requests', [])
                                review_configs = getattr(interrupt_obj, 'review_configs', [])
                                print(f"DEBUG: 1-tuple format - Actions: {len(action_requests)}, Configs: {len(review_configs)}")
                        elif len(interrupt_value) == 2:
                            # Two-element tuple: (action_requests, review_configs)
                            action_requests, review_configs = interrupt_value
                            print(f"DEBUG: 2-tuple format - Actions: {len(action_requests)}, Configs: {len(review_configs)}")
                        else:
                            # Unknown tuple format
                            action_requests = []
                            review_configs = []
                            print(f"DEBUG: Unknown tuple format with {len(interrupt_value)} elements")
                    else:
                        # Handle object format
                        action_requests = getattr(interrupt_value, 'action_requests', [])
                        review_configs = getattr(interrupt_value, 'review_configs', [])
                        print(f"DEBUG: Object format - Actions: {len(action_requests)}, Configs: {len(review_configs)}")

                    # Convert to dict for JSON serialization
                    interrupt_data = {
                        "action_requests": [],
                        "review_configs": []
                    }

                    # Extract action requests
                    for i, action in enumerate(action_requests):
                        print(f"DEBUG: Processing action {i}: {type(action)}, {action}")
                        # Handle both dict and object formats, and both 'name' and 'tool' field names
                        if isinstance(action, dict):
                            tool_name = action.get('tool') or action.get('name')
                            tool_call_id = action.get('tool_call_id', f"call_{i}")
                            args = action.get('args', {})
                            description = action.get('description')
                        else:
                            tool_name = getattr(action, 'tool', None) or getattr(action, 'name', None)
                            tool_call_id = getattr(action, 'tool_call_id', f"call_{i}")
                            args = getattr(action, 'args', {})
                            description = getattr(action, 'description', None)

                        action_dict = {
                            "tool": tool_name,
                            "tool_call_id": tool_call_id,
                            "args": args,
                            "description": description
                        }
                        print(f"DEBUG: Action dict: {action_dict}")
                        interrupt_data["action_requests"].append(action_dict)

                    # Extract review configs
                    for i, config in enumerate(review_configs):
                        print(f"DEBUG: Processing config {i}: {type(config)}, {config}")
                        config_dict = {
                            "allowed_decisions": getattr(config, 'allowed_decisions', config.get('allowed_decisions') if isinstance(config, dict) else [])
                        }
                        print(f"DEBUG: Config dict: {config_dict}")
                        interrupt_data["review_configs"].append(config_dict)

                    print(f"DEBUG: Final interrupt_data: {interrupt_data}")
                    yield {
                        "interrupt": interrupt_data,
                        "status": "interrupt"
                    }
                    continue

                # Regular update processing
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
