# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2025-11-19

### Added
- **Stop Execution Button**: Added ability to cancel ongoing agent execution with a red stop button that replaces the send button during processing
  - Backend cancellation endpoint (`/cancel`) with thread-safe execution tracking
  - Graceful cancellation between streaming chunks
  - User feedback when execution is cancelled

- **Multi-line Input Support**: Input box now supports multiple lines for longer messages
  - Enter key sends message
  - Shift+Enter creates new line
  - Manual vertical resizing with drag handle
  - Auto-constrained between 40px and 200px height

- **Non-blocking Execution**: Agent operations now run in thread pool to keep Jupyter responsive
  - Notebooks remain interactive while agent is working
  - No UI freezing during long-running operations
  - Thread pool with 4 max workers for concurrent operations

### Changed
- Professional UI redesign with light color palette
  - Blue gradient send button with hover effects
  - Red gradient stop button with professional styling
  - Refined shadows, borders, and spacing throughout
  - System messages now left-aligned with subtle gray styling

- Improved markdown rendering with compact spacing
  - Reduced paragraph padding to 0.1em vertical
  - Tighter line height for efficient reading
  - Better typography consistency

### Fixed
- Fixed raw tool call dictionaries appearing in chat output
- Fixed todo list parsing to handle Python-style single quotes using `ast.literal_eval()`
- Fixed todo list display issues after content filtering

## [0.1.0] - 2025-11-17

### Added
- Initial release
- JupyterLab extension with chat interface
- DeepAgents integration
- Real-time streaming responses
- Tool call visualization
- Todo list tracking
- Human-in-the-loop interrupts
- Context awareness (current directory and focused widget)
