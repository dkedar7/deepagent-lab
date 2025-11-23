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

    def __init__(self, agent_module_path: str = "jupyter_deepagents.my_agent", agent_variable_name: Optional[str] = None):
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
            context: Context dict with current_directory, focused_widget, selected_text, and selection_metadata

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
        if context.get("selected_text"):
            selected = context['selected_text']
            selection_metadata = context.get("selection_metadata", "")

            # Truncate very long selections to avoid token bloat
            max_length = 2000
            if len(selected) > max_length:
                selected = selected[:max_length] + f"\n... (truncated, {len(selected) - max_length} more characters)"

            # Format location information
            location_info = ""
            if selection_metadata:
                if selection_metadata.startswith("cell_"):
                    cell_idx = selection_metadata.replace("cell_", "")
                    location_info = f" from cell index {cell_idx}"
                elif selection_metadata.startswith("line_"):
                    line_num = selection_metadata.replace("line_", "")
                    location_info = f" from line {line_num}"
                elif selection_metadata.startswith("lines_"):
                    line_range = selection_metadata.replace("lines_", "")
                    location_info = f" from lines {line_range}"

            context_parts.append(f"User has selected the following text{location_info}:\n```\n{selected}\n```")

        if context_parts:
            return f"{message}\n\n" + "\n".join(context_parts)
        return message

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
                                    # Special handling for think_tool - show reflection as regular content
                                    if hasattr(last_message, 'name') and last_message.name == 'think_tool':
                                        # Extract reflection from tool message content
                                        tool_content = last_message.content
                                        reflection = None

                                        if isinstance(tool_content, str):
                                            # Try to parse as JSON
                                            try:
                                                import json
                                                parsed = json.loads(tool_content)
                                                reflection = parsed.get('reflection')
                                            except:
                                                reflection = tool_content
                                        elif isinstance(tool_content, dict):
                                            reflection = tool_content.get('reflection')

                                        if reflection:
                                            yield {
                                                "chunk": reflection,
                                                "status": "streaming"
                                            }
                                    # Special handling for write_todos - show as formatted list
                                    elif hasattr(last_message, 'name') and last_message.name == 'write_todos':
                                        tool_content = last_message.content
                                        todos = None

                                        # Try multiple parsing approaches
                                        if isinstance(tool_content, str):
                                            import json
                                            import re
                                            import ast

                                            # Look for array pattern first (handles "Updated todo list to [...]" format)
                                            match = re.search(r'\[.*\]', tool_content, re.DOTALL)
                                            if match:
                                                array_str = match.group(0)

                                                # Try parsing as Python literal first (handles single quotes)
                                                try:
                                                    todos = ast.literal_eval(array_str)
                                                except:
                                                    # Fall back to JSON parsing (requires double quotes)
                                                    try:
                                                        todos = json.loads(array_str)
                                                    except:
                                                        pass
                                            else:
                                                # No array found, try parsing entire string as JSON
                                                try:
                                                    parsed = json.loads(tool_content)
                                                    if isinstance(parsed, dict):
                                                        todos = parsed.get('todos')
                                                        # If todos is a string, parse it again
                                                        if isinstance(todos, str):
                                                            todos = json.loads(todos)
                                                    elif isinstance(parsed, list):
                                                        # Content is directly a list
                                                        todos = parsed
                                                except:
                                                    pass
                                        elif isinstance(tool_content, dict):
                                            todos = tool_content.get('todos')
                                            if isinstance(todos, str):
                                                try:
                                                    import json
                                                    todos = json.loads(todos)
                                                except:
                                                    pass
                                        elif isinstance(tool_content, list):
                                            # Content is directly a list
                                            todos = tool_content

                                        if todos and isinstance(todos, list):
                                            yield {
                                                "todo_list": todos,
                                                "status": "streaming"
                                            }
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
                                            # Skip think_tool and write_todos - shown as special content
                                            tool_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, 'name', None)
                                            if tool_name in ['think_tool', 'write_todos']:
                                                continue

                                            tool_calls.append({
                                                "id": tc.get("id") if isinstance(tc, dict) else getattr(tc, 'id', None),
                                                "name": tool_name,
                                                "args": tc.get("args") if isinstance(tc, dict) else getattr(tc, 'args', {})
                                            })

                                    content_str = content_str.strip() if content_str else ""

                                    # Always filter tool call dictionaries, even if all tool_calls were filtered out
                                    if content_str and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                                        import re
                                        tool_dict_pattern = r"\{'id':\s*'[^']+',\s*'input':\s*\{.*?\},\s*'name':\s*'[^']+',\s*'type':\s*'tool_use'\}"
                                        content_str = re.sub(tool_dict_pattern, '', content_str, flags=re.DOTALL)
                                        content_str = content_str.strip()

                                    if tool_calls and len(tool_calls) > 0:
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

                    # Handle different formats
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
                            # Unknown tuple format
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

                        action_dict = {
                            "tool": tool_name,
                            "tool_call_id": tool_call_id,
                            "args": args,
                            "description": description
                        }
                        interrupt_data["action_requests"].append(action_dict)

                    # Extract review configs
                    for i, config in enumerate(review_configs):
                        config_dict = {
                            "allowed_decisions": getattr(config, 'allowed_decisions', config.get('allowed_decisions') if isinstance(config, dict) else [])
                        }
                        interrupt_data["review_configs"].append(config_dict)
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
                                if message_type == 'ToolMessage':
                                    # Special handling for think_tool - show reflection as regular content
                                    if hasattr(last_message, 'name') and last_message.name == 'think_tool':
                                        # Extract reflection from tool message content
                                        tool_content = last_message.content
                                        reflection = None

                                        if isinstance(tool_content, str):
                                            # Try to parse as JSON
                                            try:
                                                import json
                                                parsed = json.loads(tool_content)
                                                reflection = parsed.get('reflection')
                                            except:
                                                reflection = tool_content
                                        elif isinstance(tool_content, dict):
                                            reflection = tool_content.get('reflection')

                                        if reflection:
                                            yield {
                                                "chunk": reflection,
                                                "status": "streaming"
                                            }
                                    # Special handling for write_todos - show as formatted list
                                    elif hasattr(last_message, 'name') and last_message.name == 'write_todos':
                                        tool_content = last_message.content
                                        todos = None

                                        # Try multiple parsing approaches
                                        if isinstance(tool_content, str):
                                            import json
                                            import re
                                            import ast

                                            # Look for array pattern first (handles "Updated todo list to [...]" format)
                                            match = re.search(r'\[.*\]', tool_content, re.DOTALL)
                                            if match:
                                                array_str = match.group(0)

                                                # Try parsing as Python literal first (handles single quotes)
                                                try:
                                                    todos = ast.literal_eval(array_str)
                                                except:
                                                    # Fall back to JSON parsing (requires double quotes)
                                                    try:
                                                        todos = json.loads(array_str)
                                                    except:
                                                        pass
                                            else:
                                                # No array found, try parsing entire string as JSON
                                                try:
                                                    parsed = json.loads(tool_content)
                                                    if isinstance(parsed, dict):
                                                        todos = parsed.get('todos')
                                                        # If todos is a string, parse it again
                                                        if isinstance(todos, str):
                                                            todos = json.loads(todos)
                                                    elif isinstance(parsed, list):
                                                        # Content is directly a list
                                                        todos = parsed
                                                except:
                                                    pass
                                        elif isinstance(tool_content, dict):
                                            todos = tool_content.get('todos')
                                            if isinstance(todos, str):
                                                try:
                                                    import json
                                                    todos = json.loads(todos)
                                                except:
                                                    pass
                                        elif isinstance(tool_content, list):
                                            # Content is directly a list
                                            todos = tool_content

                                        if todos and isinstance(todos, list):
                                            yield {
                                                "todo_list": todos,
                                                "status": "streaming"
                                            }
                                    # Skip other ToolMessages - don't send to frontend
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
                                            # Skip think_tool and write_todos - shown as special content
                                            tool_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, 'name', None)
                                            if tool_name in ['think_tool', 'write_todos']:
                                                continue

                                            tool_calls.append({
                                                "id": tc.get("id") if isinstance(tc, dict) else getattr(tc, 'id', None),
                                                "name": tool_name,
                                                "args": tc.get("args") if isinstance(tc, dict) else getattr(tc, 'args', {})
                                            })

                                    # Clean content_str: strip whitespace
                                    content_str = content_str.strip() if content_str else ""

                                    # Filter out tool call dictionaries from content
                                    # These often appear as strings like "{'id': '...', 'input': {...}, 'name': '...', 'type': 'tool_use'}"
                                    # Always filter, even if we filtered out all tool_calls (e.g., write_todos, think_tool)
                                    if content_str and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                                        # Remove lines that look like tool call dictionaries
                                        import re
                                        # Pattern to match tool call dictionary representations
                                        tool_dict_pattern = r"\{'id':\s*'[^']+',\s*'input':\s*\{.*?\},\s*'name':\s*'[^']+',\s*'type':\s*'tool_use'\}"
                                        content_str = re.sub(tool_dict_pattern, '', content_str, flags=re.DOTALL)
                                        content_str = content_str.strip()

                                    # Yield tool calls (if any)
                                    if tool_calls and len(tool_calls) > 0:
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
