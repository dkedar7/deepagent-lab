# deepagent-lab

A JupyterLab extension providing an AI agent chat interface with notebook manipulation capabilities and human-in-the-loop controls.

## Features

- **Chat Interface**: Sidebar for natural conversations with your agent
- **Notebook Manipulation**: Built-in tools for creating, editing, and executing Jupyter notebooks
- **Human-in-the-Loop**: Review and approve agent actions before execution
- **Context Awareness**: Automatically sends workspace and file context to your agent
- **Agent Portability**: Use any other langgraph-compatible agent seamlessly

## Installation

```bash
pip install deepagent-lab
```

## Quick Start

1. **Set up your environment** (copy `.env.example` to `.env` and configure):

```bash
# Required: Jupyter server configuration (must match your JupyterLab startup)
DEEPAGENT_JUPYTER_SERVER_URL=http://localhost:8889
DEEPAGENT_JUPYTER_TOKEN=8e2121e58cd3f9e13fc05fc020955c6e # Generate with python3 -c "import secrets; print(secrets.token_hex(16))"

# If using the default agent, Anthropic API key is required
ANTHROPIC_API_KEY=your-api-key-here

# Or if you want to use your agent, specify the location here
DEEPAGENT_AGENT_SPEC=./my_agent.py:agent
```

2. **Start JupyterLab** with matching server URL and token:

```bash
# Start JupyterLab with values matching your .env file
jupyter lab --port=8889 --IdentityProvider.token=8e2121e58cd3f9e13fc05fc020955c6e
```

**Important:** The Jupyter server URL and token in your `.env` file must match the values JupyterLab uses when starting up. This allows the agent to connect to notebook kernels for code execution.

3. **Open the chat interface** by clicking the chat icon in the right sidebar

4. **Start chatting** with your agent!

## Agent Configuration

The extension uses the **DEEPAGENT_** prefix for all environment variables, enabling full compatibility with [deepagent-dash](https://github.com/dkedar7/deepagent-dash).

### Quick Configuration

**Option 1: Use the default agent**
- The extension includes a built-in agent for notebook manipulation
- No configuration needed - just start chatting!

**Option 2: Use a custom agent**

Create a custom agent and point to it using environment variables:

```bash
# Agent spec in format "module_or_file:variable"
DEEPAGENT_AGENT_SPEC=./my_agent.py:agent
```

### Environment Variables

All configuration uses the `DEEPAGENT_` prefix for compatibility with deepagent-dash:

| Variable | Purpose | Default |
|----------|---------|---------|
| `DEEPAGENT_AGENT_SPEC` | Agent location (`path:variable`) | Uses default agent |
| `DEEPAGENT_WORKSPACE_ROOT` | Working directory for agent | JupyterLab root |
| `DEEPAGENT_MODEL_NAME` | Model identifier | `anthropic:claude-sonnet-4-20250514` |
| `DEEPAGENT_MODEL_TEMPERATURE` | Model temperature (0.0-1.0) | `0.0` |
| `DEEPAGENT_JUPYTER_SERVER_URL` | Jupyter server URL | `http://localhost:8889` |
| `DEEPAGENT_JUPYTER_TOKEN` | Jupyter auth token | `12345` |
| `DEEPAGENT_VIRTUAL_MODE` | Safe mode for filesystem | `true` |
| `DEEPAGENT_DEBUG` | Enable debug logging | `false` |

See [.env.example](.env.example) for complete configuration options.

### Creating Custom Agents

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langgraph.checkpoint.memory import MemorySaver
import os

# Agent discovers workspace automatically
workspace = os.getenv('DEEPAGENT_WORKSPACE_ROOT', '.')

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-20250514",
    backend=FilesystemBackend(root_dir=workspace, virtual_mode=True),
    checkpointer=MemorySaver(),
    tools=[...your_tools...]
)
```

Save this as `my_agent.py` and configure:
```bash
DEEPAGENT_AGENT_SPEC=./my_agent.py:agent
```

### Agent Portability

Agents configured for deepagent-lab work seamlessly in [deepagent-dash](https://github.com/dkedar7/deepagent-dash):

```bash
# Same .env file works for both!
DEEPAGENT_AGENT_SPEC=./my_agent.py:agent
DEEPAGENT_WORKSPACE_ROOT=/path/to/project

# Run in JupyterLab
jupyter lab

# Or run in Dash
deepagent-dash run
```

## Interface Controls

- **⟳ Reload**: Reload your agent without restarting JupyterLab
- **Clear**: Start a new conversation thread
- **Status Indicator**:
  - 🟢 Green: Agent ready
  - 🟠 Orange: Agent loading
  - 🔴 Red: Agent error

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.
