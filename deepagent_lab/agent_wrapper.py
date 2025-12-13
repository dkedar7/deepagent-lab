"""
Wrapper for LangGraph agent to provide a consistent API for the extension.
"""
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from dotenv import load_dotenv
load_dotenv()

# Import configuration
from . import config
from .langgraph_utils import stream_graph_updates, prepare_agent_input


class AgentWrapper:
    """Wrapper class for LangGraph agent."""

    def __init__(self, agent_module_path: Optional[str] = None, agent_variable_name: Optional[str] = None):
        """
        Initialize the agent wrapper.

        Priority for agent resolution:
        1. DEEPAGENT_AGENT_SPEC environment variable (format: "module_or_file:variable")
        2. Function parameters (agent_module_path, agent_variable_name)
        3. Default values from config module

        Args:
            agent_module_path: Path to the module or file containing the agent.
                              Can be a module path (e.g., "my_package.agent")
                              or a file path (e.g., "./my_agent.py" or "/abs/path/agent.py")
                              Defaults to config.AGENT_MODULE if not provided.
            agent_variable_name: Name of the variable to load from the module.
                                Defaults to None (will try 'agent' then 'graph').
        """
        self.agent = None

        # Check for environment variable spec
        agent_spec = config.AGENT_SPEC

        if agent_spec:
            # Parse "module_or_file:variable" format
            parts = agent_spec.split(':', 1)
            if len(parts) == 2:
                self.agent_module_path = parts[0]
                self.agent_variable_name = parts[1]
                print(f"Using agent from environment: {self.agent_module_path}:{self.agent_variable_name}")
            else:
                print(f"Warning: DEEPAGENT_AGENT_SPEC format should be 'module:variable', got: {agent_spec}")
                print(f"Falling back to parameters or defaults")
                self.agent_module_path = agent_module_path or config.AGENT_MODULE
                self.agent_variable_name = agent_variable_name or config.AGENT_VARIABLE
        else:
            # Use function parameters or config defaults
            self.agent_module_path = agent_module_path or config.AGENT_MODULE
            self.agent_variable_name = agent_variable_name or config.AGENT_VARIABLE

        self._load_agent()

    def _load_agent_from_file(self, file_path: Path, variable_name: Optional[str] = None):
        """
        Load agent from a Python file path.

        Args:
            file_path: Path to Python file containing agent
            variable_name: Optional variable name to extract

        Returns:
            Loaded agent object

        Raises:
            ImportError: If file not found or cannot be loaded
            AttributeError: If variable not found in module
        """
        file_path = file_path.resolve()

        if not file_path.exists():
            raise ImportError(f"Agent file not found: {file_path}")

        # Create a unique module name to avoid conflicts
        module_name = f"custom_agent_{file_path.stem}_{id(file_path)}"

        # Load the module from file
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to create module spec from {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Extract the agent variable
        if variable_name:
            if not hasattr(module, variable_name):
                raise AttributeError(
                    f"Module {file_path} does not have '{variable_name}' attribute"
                )
            return getattr(module, variable_name)
        else:
            # Try default names
            if hasattr(module, 'agent'):
                return module.agent
            elif hasattr(module, 'graph'):
                return module.graph
            else:
                raise AttributeError(
                    f"Module {file_path} does not have 'agent' or 'graph' attribute"
                )

    def _load_agent_from_module(self, module_path: str, variable_name: Optional[str] = None):
        """
        Load agent from a Python module path.

        Args:
            module_path: Python module path (e.g., "my_package.agent")
            variable_name: Optional variable name to extract

        Returns:
            Loaded agent object

        Raises:
            ImportError: If module cannot be imported
            AttributeError: If variable not found in module
        """
        module = importlib.import_module(module_path)

        # Extract the agent variable
        if variable_name:
            if not hasattr(module, variable_name):
                raise AttributeError(
                    f"Module {module_path} does not have '{variable_name}' attribute"
                )
            return getattr(module, variable_name)
        else:
            # Try default names
            if hasattr(module, 'agent'):
                return module.agent
            elif hasattr(module, 'graph'):
                return module.graph
            else:
                raise AttributeError(
                    f"Module {module_path} does not have 'agent' or 'graph' attribute"
                )

    def _load_agent(self):
        """
        Load the agent from the specified module or file path.

        Automatically detects whether agent_module_path is a file path or module path:
        - If it ends with .py or contains / or \\, treat as file path
        - Otherwise, treat as module path
        """
        try:
            # Determine if this is a file path or module path
            is_file_path = (
                self.agent_module_path.endswith('.py') or
                '/' in self.agent_module_path or
                '\\' in self.agent_module_path or
                self.agent_module_path.startswith('.')
            )

            if is_file_path:
                # Load from file path
                file_path = Path(self.agent_module_path)
                self.agent = self._load_agent_from_file(file_path, self.agent_variable_name)
                print(f"Loaded agent from file: {file_path}")
                if self.agent_variable_name:
                    print(f"  Variable: {self.agent_variable_name}")
            else:
                # Load from module path
                self.agent = self._load_agent_from_module(
                    self.agent_module_path,
                    self.agent_variable_name
                )
                print(f"Loaded agent from module: {self.agent_module_path}")
                if self.agent_variable_name:
                    print(f"  Variable: {self.agent_variable_name}")

        except ImportError as e:
            print(f"Warning: Could not import agent module '{self.agent_module_path}': {e}")
            print("Agent functionality will not be available until the module is created.")
            if config.AGENT_SPEC:
                print(f"Note: DEEPAGENT_AGENT_SPEC is set to: {config.AGENT_SPEC}")
            self.agent = None
        except AttributeError as e:
            print(f"Warning: {e}")
            print("Agent functionality will not be available.")
            self.agent = None
        except Exception as e:
            print(f"Error loading agent: {e}")
            if config.DEBUG:
                import traceback
                traceback.print_exc()
            self.agent = None

    def reload_agent(self):
        """Reload the agent module (useful for development)."""
        # Clear the module from sys.modules if it's there
        modules_to_remove = [
            mod_name for mod_name in sys.modules
            if (self.agent_module_path in mod_name or
                mod_name.startswith('custom_agent_'))
        ]
        for mod_name in modules_to_remove:
            del sys.modules[mod_name]

        # Reload
        self._load_agent()

    def set_root_dir(self, root_dir: str):
        """
        Set the root directory on the agent's backend if it has one.
        Also sets the DEEPAGENT_WORKSPACE_ROOT environment variable.

        Args:
            root_dir: The root directory path (JupyterLab launch directory)
        """
        # Set environment variable for agents to discover
        os.environ['DEEPAGENT_WORKSPACE_ROOT'] = root_dir

        if self.agent and hasattr(self.agent, 'backend'):
            try:
                # Import FilesystemBackend dynamically
                from deepagents.tools.filesystem import FilesystemBackend
                # Update the backend's root_dir
                self.agent.backend = FilesystemBackend(
                    root_dir=root_dir,
                    virtual_mode=config.VIRTUAL_MODE
                )
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

    def execute(
        self,
        message: Optional[str] = None,
        decisions: Optional[list] = None,
        config: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Execute the agent with either a message or resume from interrupt.

        This unified method handles both regular streaming and interrupt resumption,
        providing true decoupling from LangGraph-specific abstractions.

        Args:
            message: Optional user message to send to the agent
            decisions: Optional list of decisions to resume from interrupt
            config: Optional configuration for the agent
            thread_id: Optional thread ID for conversation history
            context: Optional context with current_directory and focused_widget

        Yields:
            Dict containing chunks of the agent's response

        Note:
            Must provide exactly one of: message or decisions
        """
        if self.agent is None:
            error_msg = "Agent not loaded. "
            if os.environ.get('JUPYTER_AGENT_PATH'):
                error_msg += f"Check JUPYTER_AGENT_PATH: {os.environ.get('JUPYTER_AGENT_PATH')}"
            else:
                error_msg += "Please create agent.py with your LangGraph agent or set JUPYTER_AGENT_PATH."
            yield {
                "error": error_msg,
                "status": "error"
            }
            return

        # Prepare config with thread_id if provided
        agent_config = config or {}
        if thread_id:
            agent_config["configurable"] = agent_config.get("configurable", {})
            agent_config["configurable"]["thread_id"] = thread_id

        try:
            # Handle message input
            if message is not None:
                # Append context to message
                message_with_context = self._append_context_to_message(message, context)
                agent_input = prepare_agent_input(message=message_with_context)
            # Handle resume from interrupt
            elif decisions is not None:
                agent_input = prepare_agent_input(decisions=decisions)
            else:
                yield {
                    "error": "Must provide either 'message' or 'decisions'",
                    "status": "error"
                }
                return

            # Stream using the unified function
            for chunk in stream_graph_updates(self.agent, agent_input, config=agent_config):
                yield chunk

        except Exception as e:
            yield {
                "error": f"Error executing agent: {str(e)}",
                "status": "error"
            }

    def resume_from_interrupt(self, decisions: list, config: Optional[Dict[str, Any]] = None, thread_id: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        """
        Resume execution after a human-in-the-loop interrupt.

        This is a convenience wrapper around execute() for backward compatibility.

        Args:
            decisions: List of decision objects with 'type' and optional fields
            config: Optional configuration for the agent
            thread_id: Thread ID to resume

        Yields:
            Dict containing chunks of the agent's response
        """
        for chunk in self.execute(decisions=decisions, config=config, thread_id=thread_id):
            yield chunk

    def stream(self, message: str, config: Optional[Dict[str, Any]] = None, thread_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
        """
        Stream responses from the agent.

        This is a convenience wrapper around execute() for backward compatibility.

        Args:
            message: The user message to send to the agent
            config: Optional configuration for the agent
            thread_id: Optional thread ID for conversation history
            context: Optional context with current_directory and focused_notebook

        Yields:
            Dict containing chunks of the agent's response
        """
        for chunk in self.execute(message=message, config=config, thread_id=thread_id, context=context):
            yield chunk


# Global agent instance
_agent_instance: Optional[AgentWrapper] = None


def get_agent() -> AgentWrapper:
    """Get or create the global agent instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = AgentWrapper()
    return _agent_instance
