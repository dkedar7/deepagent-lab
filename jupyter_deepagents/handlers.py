"""
HTTP request handlers for the DeepAgents extension.
"""
import json
from typing import Optional

from jupyter_server.base.handlers import APIHandler
from jupyter_server.utils import url_path_join
import tornado
from tornado.web import HTTPError

from .agent_wrapper import get_agent


class ChatHandler(APIHandler):
    """Handler for chat messages."""

    @tornado.web.authenticated
    async def post(self):
        """
        Handle POST requests to send a message to the agent.

        Expected JSON payload:
        {
            "message": "user message",
            "stream": false,  // optional, default false
            "thread_id": "uuid"  // optional, for conversation history
        }
        """
        try:
            # Parse request body
            data = self.get_json_body()
            message = data.get("message")
            use_stream = data.get("stream", False)
            thread_id = data.get("thread_id")
            current_directory = data.get("current_directory", "")
            focused_widget = data.get("focused_widget", "")

            if not message:
                raise HTTPError(400, "Message is required")

            # Get root directory from server settings
            root_dir = self.settings.get("server_root_dir", "")

            # Get agent instance and set root directory
            agent = get_agent()
            agent.set_root_dir(root_dir)

            # Create context object
            context = {
                "current_directory": current_directory,
                "focused_widget": focused_widget
            }

            if use_stream:
                # Stream response
                self.set_header("Content-Type", "text/event-stream")
                self.set_header("Cache-Control", "no-cache")
                self.set_header("Connection", "keep-alive")

                for chunk in agent.stream(message, thread_id=thread_id, context=context):
                    # Send as server-sent event
                    event_data = f"data: {json.dumps(chunk)}\n\n"
                    self.write(event_data)
                    await self.flush()

                self.finish()
            else:
                # Regular invoke
                result = agent.invoke(message, thread_id=thread_id, context=context)
                self.finish(json.dumps(result))

        except HTTPError:
            raise
        except Exception as e:
            self.log.error(f"Error in ChatHandler: {e}", exc_info=True)
            raise HTTPError(500, str(e))


class ReloadAgentHandler(APIHandler):
    """Handler to reload the agent module."""

    @tornado.web.authenticated
    async def post(self):
        """Reload the agent module."""
        try:
            agent = get_agent()
            agent.reload_agent()
            self.finish(json.dumps({
                "status": "success",
                "message": "Agent reloaded successfully"
            }))
        except Exception as e:
            self.log.error(f"Error reloading agent: {e}", exc_info=True)
            raise HTTPError(500, str(e))


class HealthHandler(APIHandler):
    """Handler to check if the agent is loaded."""

    @tornado.web.authenticated
    async def get(self):
        """Check agent health status."""
        try:
            agent = get_agent()
            is_loaded = agent.agent is not None

            self.finish(json.dumps({
                "status": "healthy" if is_loaded else "agent_not_loaded",
                "agent_loaded": is_loaded,
                "message": "Agent is ready" if is_loaded else "Agent module not found or failed to load"
            }))
        except Exception as e:
            self.log.error(f"Error checking health: {e}", exc_info=True)
            raise HTTPError(500, str(e))


def setup_handlers(web_app):
    """Setup the HTTP request handlers."""
    host_pattern = ".*$"
    base_url = web_app.settings["base_url"]

    # Define route patterns
    route_pattern_chat = url_path_join(base_url, "jupyter-deepagents", "chat")
    route_pattern_reload = url_path_join(base_url, "jupyter-deepagents", "reload")
    route_pattern_health = url_path_join(base_url, "jupyter-deepagents", "health")

    # Add handlers
    handlers = [
        (route_pattern_chat, ChatHandler),
        (route_pattern_reload, ReloadAgentHandler),
        (route_pattern_health, HealthHandler),
    ]

    web_app.add_handlers(host_pattern, handlers)
