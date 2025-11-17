import React, { useState, useEffect, useRef } from 'react';
import { ReactWidget } from '@jupyterlab/apputils';
import { JupyterFrontEnd } from '@jupyterlab/application';
import { IFileBrowserFactory } from '@jupyterlab/filebrowser';
import { requestAPI } from './handler';
import ReactMarkdown from 'react-markdown';

/**
 * Tool call interface
 */
interface ToolCall {
  id: string;
  name: string;
  args: Record<string, any>;
}

/**
 * Action request interface for interrupts
 */
interface ActionRequest {
  tool: string;
  tool_call_id: string;
  args: Record<string, any>;
  description?: string;
}

/**
 * Review config interface
 */
interface ReviewConfig {
  allowed_decisions: string[];
}

/**
 * Interrupt data interface
 */
interface InterruptData {
  action_requests: ActionRequest[];
  review_configs: ReviewConfig[];
}

/**
 * Decision interface
 */
interface Decision {
  type: 'approve' | 'reject' | 'edit';
  args?: Record<string, any>;
  explanation?: string;
}

/**
 * Message interface for chat messages
 */
interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  error?: boolean;
  intermediates?: string[];
  toolCalls?: ToolCall[];
  interrupt?: InterruptData;
}

/**
 * Chat component props
 */
interface ChatComponentProps {
  shell: JupyterFrontEnd.IShell | null;
  browserFactory: IFileBrowserFactory | null;
}

/**
 * Chat component
 */
const ChatComponent: React.FC<ChatComponentProps> = ({ shell, browserFactory }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [agentStatus, setAgentStatus] = useState<'unknown' | 'healthy' | 'error'>('unknown');
  const [threadId, setThreadId] = useState<string>(() => crypto.randomUUID());
  const [awaitingDecision, setAwaitingDecision] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Check agent health on mount
  useEffect(() => {
    checkAgentHealth();
  }, []);

  const checkAgentHealth = async () => {
    try {
      const response = await requestAPI<any>('health', {
        method: 'GET'
      });

      if (response.agent_loaded) {
        setAgentStatus('healthy');
        addSystemMessage('Agent is ready and connected');
      } else {
        setAgentStatus('error');
        addSystemMessage('Agent not loaded. Please ensure my_agent.py is configured correctly.');
      }
    } catch (error) {
      console.error('Error checking agent health:', error);
      setAgentStatus('error');
      addSystemMessage('Failed to connect to agent service');
    }
  };

  const addSystemMessage = (content: string) => {
    const systemMessage: Message = {
      id: Date.now().toString(),
      role: 'system',
      content,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, systemMessage]);
  };

  const getXSRFToken = (): string => {
    const matches = document.cookie.match('\\b_xsrf=([^;]*)\\b');
    return matches ? matches[1] : '';
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputValue,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    const savedInput = inputValue;
    setInputValue('');
    setIsLoading(true);

    // Get current directory and focused widget
    let currentDirectory = '';
    let focusedWidget = '';

    if (browserFactory) {
      const browser = browserFactory.tracker.currentWidget;
      if (browser) {
        currentDirectory = browser.model.path;
      }
    }

    // Get the currently focused widget from the shell
    if (shell && shell.currentWidget) {
      const current = shell.currentWidget;
      const title = current.title.label;

      // Check if it has a context (files like notebooks, text files, etc.)
      if ((current as any).context?.path) {
        focusedWidget = (current as any).context.path;
      } else if (title === 'Launcher') {
        focusedWidget = 'JupyterLab Launcher';
      } else if (title.startsWith('Terminal')) {
        focusedWidget = `Terminal: ${title}`;
      } else if (title) {
        // Generic widget with a title
        focusedWidget = title;
      }
    }

    console.log('Sending message with thread_id:', threadId);
    console.log('Current directory:', currentDirectory);
    console.log('Focused widget:', focusedWidget);

    // Create a placeholder for intermediate updates
    const assistantMessageId = (Date.now() + 1).toString();
    let intermediateMessages: string[] = [];
    let finalContent = '';
    let toolCalls: ToolCall[] = [];

    try {
      const xsrfToken = getXSRFToken();
      const response = await fetch(
        `/jupyter-deepagents/chat`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-XSRFToken': xsrfToken
          },
          body: JSON.stringify({
            message: savedInput,
            stream: true,
            thread_id: threadId,
            current_directory: currentDirectory,
            focused_widget: focusedWidget
          })
        }
      );

      if (!response.body) {
        throw new Error('No response body');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              console.log('Received SSE data:', data);

              if (data.status === 'streaming') {
                // Handle tool calls
                if (data.tool_calls) {
                  data.tool_calls.forEach((tc: ToolCall) => {
                    toolCalls.push({ ...tc });
                  });
                }

                // Handle regular content
                if (data.chunk) {
                  intermediateMessages.push(data.chunk);
                  finalContent = data.chunk; // Keep updating final content
                  console.log('Updated content:', finalContent, 'Intermediates:', intermediateMessages.length)
                }

                // Update the message in place
                setMessages(prev => {
                  const existing = prev.find(m => m.id === assistantMessageId);
                  if (existing) {
                    return prev.map(m =>
                      m.id === assistantMessageId
                        ? {
                            ...m,
                            content: finalContent,
                            intermediates: [...intermediateMessages],
                            toolCalls: toolCalls.length > 0 ? [...toolCalls] : undefined
                          }
                        : m
                    );
                  } else {
                    return [...prev, {
                      id: assistantMessageId,
                      role: 'assistant' as const,
                      content: finalContent,
                      timestamp: new Date(),
                      intermediates: [...intermediateMessages],
                      toolCalls: toolCalls.length > 0 ? [...toolCalls] : undefined
                    }];
                  }
                });
              } else if (data.status === 'interrupt') {
                // Handle interrupt - show decision UI
                setAwaitingDecision(true);
                setMessages(prev => {
                  const existing = prev.find(m => m.id === assistantMessageId);
                  if (existing) {
                    return prev.map(m =>
                      m.id === assistantMessageId
                        ? {
                            ...m,
                            interrupt: data.interrupt
                          }
                        : m
                    );
                  } else {
                    return [...prev, {
                      id: assistantMessageId,
                      role: 'assistant' as const,
                      content: '',
                      timestamp: new Date(),
                      interrupt: data.interrupt
                    }];
                  }
                });
              } else if (data.status === 'complete') {
                console.log('Stream complete. Final content:', finalContent);
                // Stream complete - keep intermediates visible
                setMessages(prev => {
                  const existing = prev.find(m => m.id === assistantMessageId);
                  if (!existing && (finalContent || toolCalls.length > 0)) {
                    // Add message if it doesn't exist but we have content or tool calls
                    return [...prev, {
                      id: assistantMessageId,
                      role: 'assistant' as const,
                      content: finalContent,
                      timestamp: new Date(),
                      intermediates: intermediateMessages.length > 0 ? [...intermediateMessages] : undefined,
                      toolCalls: toolCalls.length > 0 ? [...toolCalls] : undefined
                    }];
                  }
                  // If message exists, leave it as is (with intermediates and tool calls)
                  return prev;
                });
              } else if (data.status === 'error') {
                console.error('Stream error:', data.error);
                setMessages(prev => [...prev, {
                  id: assistantMessageId,
                  role: 'assistant' as const,
                  content: data.error || 'An error occurred',
                  timestamp: new Date(),
                  error: true
                }]);
              }
            } catch (e) {
              console.error('Error parsing SSE data:', e);
            }
          }
        }
      }
    } catch (error) {
      console.error('Error sending message:', error);

      const errorMessage: Message = {
        id: assistantMessageId,
        role: 'assistant',
        content: `Error: ${error instanceof Error ? error.message : 'Failed to send message'}`,
        timestamp: new Date(),
        error: true
      };

      setMessages(prev => {
        const existing = prev.find(m => m.id === assistantMessageId);
        if (existing) {
          return prev.map(m => m.id === assistantMessageId ? errorMessage : m);
        } else {
          return [...prev, errorMessage];
        }
      });
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleResumeFromInterrupt = async (decisions: Decision[]) => {
    setAwaitingDecision(false);
    setIsLoading(true);

    try {
      const xsrfToken = getXSRFToken();
      const response = await fetch(
        `/jupyter-deepagents/resume`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-XSRFToken': xsrfToken
          },
          body: JSON.stringify({
            decisions: decisions,
            thread_id: threadId
          })
        }
      );

      if (!response.body) {
        throw new Error('No response body');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      const assistantMessageId = (Date.now() + 1).toString();
      let intermediateMessages: string[] = [];
      let finalContent = '';
      let toolCalls: ToolCall[] = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              console.log('Resume SSE data:', data);

              if (data.status === 'streaming') {
                if (data.tool_calls) {
                  data.tool_calls.forEach((tc: ToolCall) => {
                    toolCalls.push({ ...tc });
                  });
                }

                if (data.chunk) {
                  intermediateMessages.push(data.chunk);
                  finalContent = data.chunk;
                }

                setMessages(prev => {
                  const existing = prev.find(m => m.id === assistantMessageId);
                  if (existing) {
                    return prev.map(m =>
                      m.id === assistantMessageId
                        ? {
                            ...m,
                            content: finalContent,
                            intermediates: [...intermediateMessages],
                            toolCalls: toolCalls.length > 0 ? [...toolCalls] : undefined
                          }
                        : m
                    );
                  } else {
                    return [...prev, {
                      id: assistantMessageId,
                      role: 'assistant' as const,
                      content: finalContent,
                      timestamp: new Date(),
                      intermediates: [...intermediateMessages],
                      toolCalls: toolCalls.length > 0 ? [...toolCalls] : undefined
                    }];
                  }
                });
              } else if (data.status === 'interrupt') {
                // Another interrupt
                setAwaitingDecision(true);
                setMessages(prev => {
                  const existing = prev.find(m => m.id === assistantMessageId);
                  if (existing) {
                    return prev.map(m =>
                      m.id === assistantMessageId
                        ? {
                            ...m,
                            interrupt: data.interrupt
                          }
                        : m
                    );
                  } else {
                    return [...prev, {
                      id: assistantMessageId,
                      role: 'assistant' as const,
                      content: '',
                      timestamp: new Date(),
                      interrupt: data.interrupt
                    }];
                  }
                });
              } else if (data.status === 'complete') {
                console.log('Resume complete');
              } else if (data.status === 'error') {
                console.error('Resume error:', data.error);
                setMessages(prev => [...prev, {
                  id: assistantMessageId,
                  role: 'assistant' as const,
                  content: data.error || 'An error occurred',
                  timestamp: new Date(),
                  error: true
                }]);
              }
            } catch (e) {
              console.error('Error parsing resume SSE data:', e);
            }
          }
        }
      }
    } catch (error) {
      console.error('Error resuming from interrupt:', error);
      addSystemMessage(`Error: ${error instanceof Error ? error.message : 'Failed to resume'}`);
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleReloadAgent = async () => {
    setIsLoading(true);
    try {
      await requestAPI<any>('reload', {
        method: 'POST'
      });
      addSystemMessage('Agent reloaded successfully');
      await checkAgentHealth();
    } catch (error) {
      console.error('Error reloading agent:', error);
      addSystemMessage('Failed to reload agent');
    } finally {
      setIsLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    // Generate new thread_id for fresh conversation
    setThreadId(crypto.randomUUID());
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const formatToolArgs = (args: Record<string, any>, maxLength: number = 150): string => {
    const truncated: Record<string, any> = {};
    for (const [key, value] of Object.entries(args)) {
      const strValue = JSON.stringify(value);
      if (strValue.length > maxLength) {
        truncated[key] = strValue.substring(0, maxLength) + '...';
      } else {
        truncated[key] = value;
      }
    }
    return JSON.stringify(truncated);
  };

  return (
    <div className="deepagents-chat-container">
      {/* Header */}
      <div className="deepagents-chat-header">
        <h2 className="deepagents-chat-title">Deep Agents</h2>
        <div className="deepagents-chat-controls">
          <span
            className={`deepagents-status-indicator deepagents-status-${agentStatus}`}
            title={agentStatus === 'healthy' ? 'Agent connected' : 'Agent not available'}
          />
          <button
            className="deepagents-icon-button"
            onClick={handleReloadAgent}
            disabled={isLoading}
            title="Reload agent"
          >
            ↻
          </button>
          <button
            className="deepagents-icon-button"
            onClick={clearChat}
            title="Clear chat"
          >
            🗑
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="deepagents-chat-messages">
        {messages.length === 0 ? (
          <div className="deepagents-chat-empty">
            <p>Start a conversation with your agent</p>
          </div>
        ) : (
          messages.map(message => (
            <div
              key={message.id}
              className={`deepagents-message deepagents-message-${message.role} ${
                message.error ? 'deepagents-message-error' : ''
              }`}
            >
              <div className="deepagents-message-header">
                <span className="deepagents-message-role">
                  {message.role === 'user' ? 'You' : message.role === 'assistant' ? 'Agent' : 'System'}
                </span>
                <span className="deepagents-message-time">
                  {formatTime(message.timestamp)}
                </span>
              </div>
              <div className="deepagents-message-content">
                {message.intermediates && message.intermediates.length > 1 && (
                  <div className="deepagents-message-intermediates">
                    {message.intermediates.slice(0, -1).map((intermediate, idx) => (
                      <div key={idx} className="deepagents-intermediate-message">
                        {intermediate}
                      </div>
                    ))}
                  </div>
                )}
                {message.toolCalls && message.toolCalls.length > 0 && (
                  <div className="deepagents-tool-calls">
                    {message.toolCalls.map((toolCall, idx) => (
                      <details key={idx} className="deepagents-tool-call">
                        <summary className="deepagents-tool-call-summary">
                          {toolCall.name}
                        </summary>
                        <div className="deepagents-tool-call-args">
                          {formatToolArgs(toolCall.args)}
                        </div>
                      </details>
                    ))}
                  </div>
                )}
                {message.interrupt && (
                  <div className="deepagents-interrupt">
                    <div className="deepagents-interrupt-header">
                      Human approval required
                    </div>
                    <div className="deepagents-interrupt-description">
                      Tool: <strong>{message.interrupt.action_requests[0]?.tool}</strong>
                    </div>
                    <div className="deepagents-action-decisions">
                      {message.interrupt?.review_configs[0]?.allowed_decisions.includes('approve') && (
                        <button
                          className="deepagents-decision-btn deepagents-approve-btn"
                          onClick={() => {
                            const decisions: Decision[] = [{ type: 'approve' }];
                            handleResumeFromInterrupt(decisions);
                          }}
                          disabled={!awaitingDecision || isLoading}
                        >
                          Approve
                        </button>
                      )}
                      {message.interrupt?.review_configs[0]?.allowed_decisions.includes('reject') && (
                        <button
                          className="deepagents-decision-btn deepagents-reject-btn"
                          onClick={() => {
                            const decisions: Decision[] = [{ type: 'reject' }];
                            handleResumeFromInterrupt(decisions);
                          }}
                          disabled={!awaitingDecision || isLoading}
                        >
                          Reject
                        </button>
                      )}
                      {message.interrupt?.review_configs[0]?.allowed_decisions.includes('edit') && (
                        <button
                          className="deepagents-decision-btn deepagents-edit-btn"
                          onClick={() => {
                            const newArgsStr = prompt(
                              'Edit arguments (JSON):',
                              JSON.stringify(message.interrupt!.action_requests[0].args, null, 2)
                            );
                            if (newArgsStr) {
                              try {
                                const newArgs = JSON.parse(newArgsStr);
                                const decisions: Decision[] = [{
                                  type: 'edit',
                                  args: newArgs
                                }];
                                handleResumeFromInterrupt(decisions);
                              } catch (e) {
                                alert('Invalid JSON');
                              }
                            }
                          }}
                          disabled={!awaitingDecision || isLoading}
                        >
                          Edit
                        </button>
                      )}
                    </div>
                  </div>
                )}
                {message.role === 'assistant' ? (
                  <ReactMarkdown>{message.content}</ReactMarkdown>
                ) : (
                  message.content
                )}
              </div>
            </div>
          ))
        )}
        {isLoading && (
          <div className="deepagents-message deepagents-message-assistant">
            <div className="deepagents-message-header">
              <span className="deepagents-message-role">Agent</span>
            </div>
            <div className="deepagents-message-content deepagents-typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="deepagents-chat-input-container">
        <input
          ref={inputRef}
          type="text"
          className="deepagents-chat-input"
          placeholder="Type your message..."
          value={inputValue}
          onChange={e => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
          disabled={isLoading}
        />
        <button
          className="deepagents-send-button"
          onClick={handleSendMessage}
          disabled={!inputValue.trim() || isLoading}
          title="Send message"
        >
          ↑
        </button>
      </div>
    </div>
  );
};

/**
 * A Lumino Widget that wraps a ChatComponent.
 */
export class ChatWidget extends ReactWidget {
  private shell: JupyterFrontEnd.IShell | null;
  private browserFactory: IFileBrowserFactory | null;

  constructor(
    shell: JupyterFrontEnd.IShell | null = null,
    browserFactory: IFileBrowserFactory | null = null
  ) {
    super();
    this.addClass('deepagents-chat-widget');
    this.shell = shell;
    this.browserFactory = browserFactory;
  }

  render(): JSX.Element {
    return <ChatComponent
      shell={this.shell}
      browserFactory={this.browserFactory}
    />;
  }
}
