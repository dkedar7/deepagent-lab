from dotenv import load_dotenv
load_dotenv()

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents.middleware import HumanInTheLoopMiddleware

# Define tools
from typing import Annotated
from jupyter_client import BlockingKernelClient, find_connection_file
import nbformat
import requests
import os

kernel_clients = {}

def get_notebook_kernel_id(notebook_path: str) -> str:
    """Get kernel ID for a running notebook via Jupyter Server API."""
    server_url = 'http://localhost:8889'
    token = os.getenv('JUPYTER_TOKEN', '12345')
    
    response = requests.get(
        f'{server_url}/api/sessions',
        headers={'Authorization': f'token {token}'} if token else {}
    )
    
    if response.status_code != 200:
        raise ValueError(f"Cannot connect to Jupyter server at {server_url}")
    
    sessions = response.json()
    for session in sessions:
        if notebook_path in session['notebook']['path']:
            return session['kernel']['id']
    
    raise ValueError(f"No running kernel found for {notebook_path}")


def create_notebook(notebook_path: str) -> str:
    """Create a new empty Jupyter notebook file and return confirmation."""

    notebook_path = notebook_path.strip("/")
    nb = nbformat.v4.new_notebook()
    nbformat.write(nb, notebook_path)
    return f"Created new notebook at {notebook_path}"


def insert_code_cell(
    code: Annotated[str, "Python code for the cell"],
    notebook_path: Annotated[str, "Notebook filename"],
    position: Annotated[int, "Index to insert cell at (-1 for append)"] = -1
) -> str:
    """Insert a new code cell into the notebook file."""

    notebook_path = notebook_path.strip("/")
    
    nb = nbformat.read(notebook_path, as_version=4)
    new_cell = nbformat.v4.new_code_cell(source=code)
    new_cell.metadata['jupyter'] = {'source_hidden': True}
    
    if position == -1:
        nb.cells.append(new_cell)
        cell_idx = len(nb.cells) - 1
    else:
        nb.cells.insert(position, new_cell)
        cell_idx = position
    
    nbformat.write(nb, notebook_path)
    return f"Inserted code cell at index {cell_idx} in {notebook_path}"

def modify_cell(
    notebook_path: Annotated[str, "Notebook filename"],
    cell_index: Annotated[int, "Index of cell to modify"],
    new_code: Annotated[str, "New code (empty string to delete cell)"]
) -> str:
    """Modify or delete an existing cell in the notebook."""

    notebook_path = notebook_path.strip("/")
    
    nb = nbformat.read(notebook_path, as_version=4)
    
    if cell_index < 0 or cell_index >= len(nb.cells):
        return f"Error: Cell index {cell_index} out of range (0-{len(nb.cells)-1})"
    
    # Delete cell if new_code is empty
    if new_code == "":
        removed_cell = nb.cells.pop(cell_index)
        nbformat.write(nb, notebook_path)
        return f"Deleted cell at index {cell_index} in {notebook_path}"
    
    # Modify cell
    cell = nb.cells[cell_index]
    cell.metadata['jupyter'] = {'source_hidden': True}
    if cell.cell_type != 'code':
        return f"Error: Cell {cell_index} is not a code cell"
    
    cell.source = new_code
    cell.outputs = []  # Clear outputs when modifying
    cell.execution_count = None  # Reset execution count
    
    nbformat.write(nb, notebook_path)
    return f"Modified cell at index {cell_index} in {notebook_path}"

def execute_cell(
    notebook_path: Annotated[str, "Notebook filename"],
    cell_index: Annotated[int, "Index of cell to execute"]
) -> str:
    """Execute a cell in the notebook kernel and update outputs in the file."""

    notebook_path = notebook_path.strip("/")
    
    # Read notebook
    nb = nbformat.read(notebook_path, as_version=4)
    cell = nb.cells[cell_index]
    
    if cell.cell_type != 'code':
        return f"Cell {cell_index} is not a code cell"
    
    # Connect to kernel
    if notebook_path not in kernel_clients:
        kernel_id = get_notebook_kernel_id(notebook_path)
        connection_file = find_connection_file(kernel_id)
        client = BlockingKernelClient()
        client.load_connection_file(connection_file)
        client.start_channels()
        kernel_clients[notebook_path] = client
    
    client = kernel_clients[notebook_path]
    msg_id = client.execute(cell.source)
    
    # Collect outputs and execution count
    outputs = []
    execution_count = None
    output_texts = []
    
    while True:
        try:
            msg = client.get_iopub_msg(timeout=5)
            if msg['parent_header'].get('msg_id') != msg_id:
                continue
            
            msg_type = msg['header']['msg_type']
            content = msg['content']
            
            if msg_type == 'execute_input':
                execution_count = content['execution_count']
            elif msg_type == 'stream':
                output = nbformat.v4.new_output('stream', 
                    name=content['name'], text=content['text'])
                outputs.append(output)
                output_texts.append(f"[{content['name']}] {content['text']}")
            elif msg_type == 'execute_result':
                output = nbformat.v4.new_output('execute_result',
                    data=content['data'], execution_count=content['execution_count'])
                outputs.append(output)
                output_texts.append(content['data'].get('text/plain', str(content['data'])))
            elif msg_type == 'display_data':
                output = nbformat.v4.new_output('display_data', data=content['data'])
                outputs.append(output)
                output_texts.append(f"[display] {content['data'].get('text/plain', 'Rich content')}")
            elif msg_type == 'error':
                output = nbformat.v4.new_output('error',
                    ename=content['ename'], evalue=content['evalue'], 
                    traceback=content['traceback'])
                outputs.append(output)
                error_msg = f"ERROR: {content['ename']}: {content['evalue']}\n" + '\n'.join(content['traceback'])
                output_texts.append(error_msg)
            elif msg_type == 'status' and content['execution_state'] == 'idle':
                break
        except:
            break
    
    # Update cell in notebook
    cell.execution_count = execution_count
    cell.outputs = outputs
    nbformat.write(nb, notebook_path)
    
    output_summary = '\n'.join(output_texts) if output_texts else "(no output)"
    return f"Executed cell [{execution_count}] in {notebook_path}:\n{output_summary}"


# Build the deep agent
system_prompt = """You're a JupyterLab assistant. Use the provided tools to manipulate and execute code cells in the specified notebook files as per user instructions.

# Guidelines:
- When asked a question, use the tool `write_todos` to plan your approach.
- If necessary, write code to answer user questions.
- Always write the requested code into a Jupyter notebook using the tools described below.
- You may choose to create a temporary Jupyter notebook file for intermediate steps.

# Tools for writing Jupyter Notebooks:
- `create_notebook(notebook_path: str) -> str`: Creates a new empty Jupyter notebook file at the specified path. Returns a confirmation message.
- `insert_code_cell(code: str, notebook_path: str, position: int = -1) -> str`: Inserts a new code cell with the given code into the specified notebook at the given position (default is to append at the end). Returns the index of the inserted cell.
- `modify_cell(notebook_path: str, cell_index: int, new_code: str) -> str`: Modifies the code of the cell at the specified index in the notebook. If `new_code` is an empty string, deletes the cell. Returns a confirmation message.
- `execute_cell(notebook_path: str, cell_index: int) -> str`: Executes the code cell at the specified index in the notebook and updates its outputs. Returns the execution result or error message.

# Examples:
User: Please create a new notebook "example.ipynb" and add a code cell that prints "Hello, World!", then execute it.
Assistant:
1. create_notebook("example.ipynb")
2. insert_code_cell('print("Hello, World!")', "example.ipynb")
3. execute_cell("example.ipynb", 0)

# Code execution:
- Always first create a new notebook if it doesn't exist.
- Insert code cells as needed and immediately execute them to verify correctness.
- Modify existing cells if corrections are needed to avoid errors.
- Ask the user for clarification if instructions are ambiguous or if you need help.
- NEVER run risky or harmful code without explicit user consent.
"""


agent = create_deep_agent(
    system_prompt=system_prompt,
    backend=FilesystemBackend(root_dir=".", virtual_mode=True),
    checkpointer=MemorySaver(),
    tools=[insert_code_cell, modify_cell, execute_cell]
)