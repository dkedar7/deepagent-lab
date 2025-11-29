# Agent Configuration Guide

## Overview

The Jupyter DeepAgents extension supports two ways to specify your agent:

1. **Default**: Place `agent.py` in your working directory
2. **Custom**: Use the `JUPYTER_AGENT_PATH` environment variable

## Method 1: Default Location (agent.py)

### Quick Start

1. Create `agent.py` in the directory where you run `jupyter lab`:

```python
from langgraph import StateGraph
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, AIMessage

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "Messages"]

def agent_logic(state: AgentState) -> AgentState:
    # Your agent logic here
    messages = state["messages"]
    last_message = messages[-1]
    response = AIMessage(content=f"You said: {last_message.content}")
    return {"messages": messages + [response]}

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_logic)
workflow.set_entry_point("agent")
workflow.add_edge("agent", "__end__")

# Export as 'agent' (or 'graph')
agent = workflow.compile()
```

2. Run `jupyter lab` from the same directory
3. The extension automatically loads your agent

### Variable Names

The extension looks for these variable names in order:
1. `agent` (preferred)
2. `graph` (alternative)

**Example:**
```python
# Either of these works:
agent = workflow.compile()  # ✅ Preferred

# Or:
graph = workflow.compile()  # ✅ Also works
```

## Method 2: Custom Location (JUPYTER_AGENT_PATH)

### Format

```bash
export JUPYTER_AGENT_PATH="module_path:variable_name"
```

- `module_path`: Python import path (e.g., `my_module` or `package.submodule`)
- `variable_name`: Name of the variable containing your compiled agent

### Examples

#### Example 1: Custom filename

```bash
# Agent in custom_agent.py
export JUPYTER_AGENT_PATH="custom_agent:agent"
jupyter lab
```

#### Example 2: Package structure

```bash
# Agent in src/agents/notebook_agent.py
export JUPYTER_AGENT_PATH="src.agents.notebook_agent:agent"
jupyter lab
```

File structure:
```
project/
├── src/
│   └── agents/
│       ├── __init__.py
│       └── notebook_agent.py  # Contains 'agent = workflow.compile()'
└── notebooks/
```

#### Example 3: Custom variable name

```bash
# Agent exported with a different name
export JUPYTER_AGENT_PATH="my_agent:my_custom_graph"
jupyter lab
```

In `agent.py`:
```python
# Export with custom name
my_custom_graph = workflow.compile()
```

#### Example 4: Multiple projects

```bash
# Project 1
cd ~/project1
export JUPYTER_AGENT_PATH="agents.project1:agent"
jupyter lab

# Project 2
cd ~/project2
export JUPYTER_AGENT_PATH="agents.project2:agent"
jupyter lab
```

### Setting Permanently

#### Option A: Shell configuration

Add to `~/.bashrc`, `~/.zshrc`, or `~/.bash_profile`:

```bash
export JUPYTER_AGENT_PATH="my_agents.main:agent"
```

Then reload:
```bash
source ~/.bashrc  # or ~/.zshrc
```

#### Option B: Jupyter configuration

Create or edit `~/.jupyter/jupyter_lab_config.py`:

```python
import os
os.environ['JUPYTER_AGENT_PATH'] = 'my_agents.main:agent'
```

#### Option C: Project-specific (.env file)

Use `python-dotenv` to load from a `.env` file:

1. Install: `pip install python-dotenv`

2. Create `.env` in your project:
```bash
JUPYTER_AGENT_PATH=agents.notebook:agent
```

3. In your Jupyter config or startup script:
```python
from dotenv import load_dotenv
load_dotenv()
```

## How It Works

### Load Order

1. Check for `JUPYTER_AGENT_PATH` environment variable
2. If set, parse as `module:variable`
3. Import the module
4. Get the specified variable
5. If not set, fall back to `agent.py` with default variable names

### Verification

When JupyterLab starts, check the terminal output:

**With environment variable:**
```
Using agent from environment: custom_agent:my_graph
Loaded agent from custom_agent.my_graph
```

**Without environment variable:**
```
Loaded agent from agent.agent
```

**Error (module not found):**
```
Warning: Could not import agent module 'custom_agent': No module named 'custom_agent'
Note: JUPYTER_AGENT_PATH is set to: custom_agent:my_graph
```

## Troubleshooting

### Agent not loading

**Check the environment variable:**
```bash
echo $JUPYTER_AGENT_PATH
```

**Verify the module can be imported:**
```bash
python -c "import sys; print(sys.path)"
python -c "from custom_agent import my_graph; print('Success')"
```

**Check JupyterLab logs:**
Look for messages like:
- `Loaded agent from ...`
- `Warning: Could not import agent module ...`

### Module not found

**Solution 1: Add to Python path**
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/your/modules"
export JUPYTER_AGENT_PATH="custom_agent:agent"
jupyter lab
```

**Solution 2: Install as package**
```bash
cd /path/to/your/agent
pip install -e .
```

Then:
```bash
export JUPYTER_AGENT_PATH="your_package.agent:agent"
jupyter lab
```

### Variable not found

**Check variable name:**
```bash
python -c "from agent import agent; print(type(agent))"
```

**Common mistakes:**
```python
# ❌ Wrong - not exported
def create_agent():
    return workflow.compile()

# ✅ Correct - exported at module level
agent = workflow.compile()
```

### Format error

**Correct format:**
```bash
export JUPYTER_AGENT_PATH="module:variable"  # ✅
```

**Incorrect formats:**
```bash
export JUPYTER_AGENT_PATH="module.variable"  # ❌ (dot instead of colon)
export JUPYTER_AGENT_PATH="module"           # ❌ (missing variable)
export JUPYTER_AGENT_PATH=":variable"        # ❌ (missing module)
```

## Best Practices

### 1. Use absolute imports

```python
# ✅ Good
from src.agents.notebook import agent

# ❌ Avoid
from ..agents.notebook import agent  # Relative import
```

### 2. Keep agent code separate

```
project/
├── agents/
│   └── notebook_agent.py  # Agent logic
├── tools/
│   └── notebook_tools.py  # Tools
└── notebooks/
    └── analysis.ipynb
```

### 3. Use descriptive variable names

```python
# ✅ Clear
notebook_assistant = workflow.compile()

# Then:
# export JUPYTER_AGENT_PATH="agents.notebook:notebook_assistant"
```

### 4. Document your setup

Include a README.md in your project:

```markdown
## Setup

Set the agent:
\`\`\`bash
export JUPYTER_AGENT_PATH="agents.notebook:agent"
jupyter lab
\`\`\`
```

### 5. Version control

Add to `.gitignore`:
```
.env
__pycache__/
*.pyc
```

Include `.env.example`:
```bash
# .env.example
JUPYTER_AGENT_PATH=agents.notebook:agent
```

## Examples by Use Case

### Academic Research
```bash
export JUPYTER_AGENT_PATH="research.agents.literature_review:agent"
```

### Data Analysis
```bash
export JUPYTER_AGENT_PATH="analytics.agents.data_explorer:agent"
```

### Development
```bash
export JUPYTER_AGENT_PATH="dev.agents.code_assistant:agent"
```

### Multi-tenant (different agents per project)
```bash
# Client A
cd ~/clients/client_a
export JUPYTER_AGENT_PATH="client_a.agent:agent"
jupyter lab --port=8888

# Client B
cd ~/clients/client_b
export JUPYTER_AGENT_PATH="client_b.agent:agent"
jupyter lab --port=8889
```

## See Also

- [README.md](README.md) - Main documentation
- [USAGE.md](USAGE.md) - Usage guide
- [agent.py](agent.py) - Example agent template
