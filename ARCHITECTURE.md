# Architecture Documentation

## Overview

The deepagent-lab extension consists of two main components:

1. **Frontend**: A JupyterLab extension (TypeScript/React) that provides the chat UI
2. **Backend**: A Jupyter Server extension (Python) that wraps and exposes your agent

## Component Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    JupyterLab UI                        │
│  ┌────────────────────────────────────────────────┐    │
│  │         Chat Widget (React)                     │    │
│  │  - Message display                              │    │
│  │  - Input field                                  │    │
│  │  - Status indicators                            │    │
│  └────────────────┬───────────────────────────────┘    │
│                   │                                      │
│                   │ HTTP/REST API                        │
│                   │                                      │
│  ┌────────────────┴───────────────────────────────┐    │
│  │       Jupyter Server Extension                  │    │
│  │  ┌──────────────────────────────────────┐      │    │
│  │  │  API Handlers                         │      │    │
│  │  │  - /chat (POST)                       │      │    │
│  │  │  - /health (GET)                      │      │    │
│  │  │  - /reload (POST)                     │      │    │
│  │  └──────────────┬───────────────────────┘      │    │
│  │                 │                               │    │
│  │  ┌──────────────┴───────────────────────┐      │    │
│  │  │  Agent Wrapper                        │      │    │
│  │  │  - Load agent module                  │      │    │
│  │  │  - invoke() / stream() interface      │      │    │
│  │  └──────────────┬───────────────────────┘      │    │
│  │                 │                               │    │
│  │  ┌──────────────┴───────────────────────┐      │    │
│  │  │  Your Agent (agent.py)             │      │    │
│  │  │  - Graph definition                   │      │    │
│  │  │  - Tools (notebook, filesystem, etc.) │      │    │
│  │  └──────────────────────────────────────┘      │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## Frontend Components

### 1. Chat Widget ([src/widget.tsx](src/widget.tsx))

The main React component that renders the chat interface. Features:

- **Message Management**: Maintains message history with roles (user, assistant, system)
- **Auto-scrolling**: Automatically scrolls to newest messages
- **Status Indicators**: Shows agent connection status
- **Loading States**: Displays typing indicators while waiting for responses
- **Error Handling**: Shows error messages in the chat

Key functions:
- `handleSendMessage()`: Sends user messages to the backend
- `checkAgentHealth()`: Verifies agent is loaded and ready
- `handleReloadAgent()`: Triggers agent module reload

### 2. Extension Entry Point ([src/index.ts](src/index.ts))

Registers the extension with JupyterLab:

- Adds command to open chat widget
- Registers widget in right sidebar
- Adds command palette entry
- Adds launcher item

### 3. API Handler ([src/handler.ts](src/handler.ts))

Utility function for making requests to the backend API:

- Handles URL construction
- Manages authentication
- Parses JSON responses
- Error handling

## Backend Components

### 1. Agent Wrapper ([deepagent_lab/agent_wrapper.py](deepagent_lab/agent_wrapper.py))

Wraps your agent to provide a consistent interface:

- **Loading**: Dynamically imports `agent.py` module
- **Invoke**: Calls `agent.invoke()` with proper input format
- **Stream**: Calls `agent.stream()` for streaming responses
- **Reload**: Supports hot-reloading of agent module during development
- **Error Handling**: Provides helpful error messages

The wrapper expects your agent to:
- Be exported as `agent` or `graph` from `agent.py`
- Accept input in the format: `{"messages": [...]}`
- Return output containing a `messages` field

### 2. HTTP Handlers ([deepagent_lab/handlers.py](deepagent_lab/handlers.py))

Defines three API endpoints:

#### POST /deepagent-lab/chat
Send a message to the agent.

Request:
```json
{
  "message": "Your message here",
  "stream": false
}
```

Response (non-streaming):
```json
{
  "response": "Agent response",
  "status": "success",
  "full_result": {...}
}
```

#### GET /deepagent-lab/health
Check if agent is loaded and ready.

Response:
```json
{
  "status": "healthy",
  "agent_loaded": true,
  "message": "Agent is ready"
}
```

#### POST /deepagent-lab/reload
Reload the agent module (useful during development).

Response:
```json
{
  "status": "success",
  "message": "Agent reloaded successfully"
}
```

### 3. Extension Initialization ([deepagent_lab/__init__.py](deepagent_lab/__init__.py))

Registers the server extension with Jupyter:

- Defines extension entry points
- Sets up HTTP handlers
- Configures labextension path

## Data Flow

### Sending a Message

1. User types message in chat input
2. React component calls `handleSendMessage()`
3. Frontend makes POST request to `/deepagent-lab/chat`
4. `ChatHandler` receives request
5. `AgentWrapper.invoke()` is called
6. Agent processes message
7. Response is formatted and sent back
8. Frontend displays response in chat

### Streaming Responses (Optional)

1. User sends message with `stream: true`
2. Backend uses Server-Sent Events (SSE)
3. `AgentWrapper.stream()` yields chunks
4. Each chunk sent as SSE event
5. Frontend can process chunks as they arrive

## Configuration

### Frontend Configuration

Located in [package.json](package.json):
- Dependencies versions
- Build scripts
- JupyterLab extension metadata

### Backend Configuration

Located in [pyproject.toml](pyproject.toml):
- Python dependencies
- Extension metadata
- Build configuration

### TypeScript Configuration

Located in [tsconfig.json](tsconfig.json):
- Compiler options
- Module resolution
- Output settings

## Customization

### Changing Agent Input/Output Format

If your agent uses a different format than `{"messages": [...]}`, modify:

1. [deepagent_lab/agent_wrapper.py](deepagent_lab/agent_wrapper.py):
   - Update `invoke()` method to format input correctly
   - Update response extraction logic

### Adding New API Endpoints

1. Add handler class in [deepagent_lab/handlers.py](deepagent_lab/handlers.py)
2. Register route in `setup_handlers()`
3. Add corresponding frontend function in [src/handler.ts](src/handler.ts)

### Styling the Chat Interface

Modify [style/base.css](style/base.css) to change:
- Colors and theming
- Layout and spacing
- Message appearance
- Button styles

All styles use JupyterLab CSS variables for theme consistency.

## Development Workflow

1. Make changes to TypeScript files in `src/`
2. Run `jlpm watch` to auto-rebuild
3. Make changes to Python files in `deepagent_lab/`
4. Use reload button in chat UI to reload agent
5. Refresh JupyterLab to see frontend changes

## Error Handling

### Frontend Errors
- Network errors: Shown as error messages in chat
- Invalid responses: Logged to browser console
- Agent unavailable: Status indicator turns red

### Backend Errors
- Import errors: Agent wrapper provides helpful message
- Agent errors: Caught and returned as error response
- HTTP errors: Standard Jupyter error handling

## Security Considerations

1. **Authentication**: All endpoints require Jupyter authentication
2. **Input Validation**: Request data is validated before processing
3. **Error Messages**: Sensitive information not exposed in errors
4. **Sandboxing**: Agent runs in same Python environment as Jupyter

## Performance

- **Lazy Loading**: Agent module loaded on first request
- **Connection Pooling**: Uses Jupyter's existing connection management
- **Efficient Rendering**: React virtual DOM for message updates
- **Auto-scroll Optimization**: Uses `requestAnimationFrame`

## Testing

To test the extension:

1. Install in development mode
2. Create a simple test agent in `agent.py`
3. Start JupyterLab
4. Open chat interface
5. Verify:
   - Status indicator shows green
   - Messages send and receive correctly
   - Error handling works
   - Reload functionality works
