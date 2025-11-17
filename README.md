# jupyter-deepagents

A JupyterLab extension that provides an elegant chat interface for AI agents with human-in-the-loop capabilities.

## Features

### Core Functionality
- **Chat Interface**: Clean, accessible sidebar interface for natural conversations with your agent
- **Streaming Responses**: Real-time streaming of agent responses for immediate feedback
- **Thread-based History**: Maintains conversation context across messages with persistent thread IDs
- **Context Awareness**: Automatically sends current directory and focused widget information to the agent

### Human-in-the-Loop
- **Tool Call Approvals**: Review and approve/reject/edit tool calls before execution
- **Minimal Design**: Simple gray UI with one-click approval buttons
- **Flexible Decisions**: Approve, reject, or edit tool arguments inline
- **No Interruptions**: Streamlined workflow without unnecessary prompts

### Developer Experience
- **Tool Call Visibility**: Expandable sections showing tool names and arguments
- **Markdown Rendering**: Compact, elegant formatting for agent responses
- **Agent Health Status**: Visual indicator showing agent connection status
- **Hot Reload**: Reload agent configuration without restarting JupyterLab
- **Clear Chat**: Start fresh conversations with a single click

## Requirements

- JupyterLab >= 4.0.0
- Python >= 3.8

## Installation

```bash
pip install jupyter-deepagents
```

## Usage

### Quick Start

1. **Create your agent** in `my_agent.py`:

```python
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver

# Create agent with interrupt capability
agent = create_deep_agent(
    backend=FilesystemBackend(root_dir=".", virtual_mode=True),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "write_file": {"allowed_decisions": ["approve", "reject"]}
            },
            description_prefix="Tool execution pending approval",
        ),
    ],
    checkpointer=MemorySaver()
)
```

2. **Start JupyterLab**:

```bash
jupyter lab
```

3. **Open the chat interface** by clicking the chat icon in the right sidebar

4. **Start chatting** with your agent!

### Using the Interface

**Basic Chat:**
- Type your message in the input field
- Press Enter or click the send arrow (↑) to send
- Watch as the agent streams its response in real-time

**Human-in-the-Loop Approvals:**

When your agent wants to execute a tool:
1. An approval UI appears showing the tool name
2. Click **Approve** to allow execution
3. Click **Reject** to deny the action
4. Click **Edit** to modify tool arguments before execution

No confirmation dialogs or reason prompts - just one click!

**Interface Controls:**
- **⟳ Reload**: Reload your agent without restarting JupyterLab
- **Clear**: Start a new conversation thread
- **Status Indicator**:
  - 🟢 Green: Agent loaded and ready
  - 🟠 Orange: Agent not found or loading
  - 🔴 Red: Agent error

### Agent Configuration

#### Option 1: Default Location (Recommended)

Create `my_agent.py` in your working directory with an `agent` variable:

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=your_model,
    tools=your_tools,
    checkpointer=MemorySaver()
)
```

#### Option 2: Custom Location

Set the `JUPYTER_AGENT_PATH` environment variable:

```bash
export JUPYTER_AGENT_PATH="path.to.module:variable_name"
jupyter lab
```

**Examples:**
```bash
# Agent in custom_agent.py as 'my_graph'
export JUPYTER_AGENT_PATH="custom_agent:my_graph"

# Agent in package: src/agents/main.py as 'agent'
export JUPYTER_AGENT_PATH="src.agents.main:agent"
```

Format: `module_path:variable_name`
- `module_path`: Python import path (e.g., `my_agent` or `package.module`)
- `variable_name`: Name of the agent variable in the module

See [AGENT_CONFIGURATION.md](AGENT_CONFIGURATION.md) for advanced configuration.


## UI Design Philosophy

The interface follows a minimal, functional aesthetic:

- **Compact Markdown**: Tight line spacing (1.2) and minimal margins for efficient reading
- **Minimal Color Scheme**: Gray, black, and white palette without visual clutter
- **No Rounded Edges**: Clean, professional design
- **One-Click Actions**: Approval buttons submit immediately without confirmation dialogs
- **Elegant Send Button**: Circular arrow button (↑) for a modern look
- **Expandable Tool Calls**: Collapsible sections keep the interface clean

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.
