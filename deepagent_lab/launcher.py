#!/usr/bin/env python3
"""
DeepAgent Lab launcher script.

This script wraps the 'jupyter lab' command to automatically configure
the Jupyter server settings and make them available to agents.

Usage:
    deepagent-lab [jupyter lab args...]

Example:
    deepagent-lab --port 8889
    deepagent-lab --no-browser
"""
import os
import sys
import socket
import secrets
import subprocess


def find_available_port(start_port=8888, max_attempts=10):
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find available port in range {start_port}-{start_port + max_attempts}")


def generate_token():
    """Generate a secure random token for Jupyter authentication."""
    return secrets.token_urlsafe(32)


def main():
    """Main launcher function."""
    # Parse command line arguments
    args = sys.argv[1:]

    # Check if user specified a port
    user_port = None
    port_specified = False
    for i, arg in enumerate(args):
        if arg == '--port' and i + 1 < len(args):
            try:
                user_port = int(args[i + 1])
                port_specified = True
            except ValueError:
                pass
            break
        elif arg.startswith('--port='):
            try:
                user_port = int(arg.split('=')[1])
                port_specified = True
            except ValueError:
                pass
            break

    # Find available port
    if user_port:
        port = user_port
        print(f"Using user-specified port: {port}")
    else:
        port = find_available_port()
        print(f"Auto-detected available port: {port}")

    # Generate token (or use existing if set)
    token = os.getenv('JUPYTER_TOKEN')
    if not token:
        token = generate_token()
        print(f"Generated secure authentication token")
    else:
        print(f"Using existing JUPYTER_TOKEN from environment")

    # Determine server URL
    # Use localhost for security (only local connections)
    server_url = f"http://localhost:{port}"

    # Set environment variables for the agent to use
    os.environ['DEEPAGENT_JUPYTER_SERVER_URL'] = server_url
    os.environ['DEEPAGENT_JUPYTER_TOKEN'] = token

    print(f"\n{'='*60}")
    print(f"DeepAgent Lab Configuration:")
    print(f"  Server URL: {server_url}")
    print(f"  Token: {'*' * 20} (hidden for security)")
    print(f"  Environment variables set:")
    print(f"    - DEEPAGENT_JUPYTER_SERVER_URL")
    print(f"    - DEEPAGENT_JUPYTER_TOKEN")
    print(f"{'='*60}\n")

    # Build jupyter lab command
    jupyter_args = ['jupyter', 'lab']

    # Add port if not already specified by user
    if not port_specified:
        jupyter_args.extend(['--port', str(port)])

    # Add token
    jupyter_args.extend(['--IdentityProvider.token', token])

    # Add any user-provided arguments
    jupyter_args.extend(args)

    # Launch Jupyter Lab
    print(f"Launching: {' '.join(jupyter_args)}\n")
    try:
        subprocess.run(jupyter_args, env=os.environ)
    except KeyboardInterrupt:
        print("\n\nShutting down DeepAgent Lab...")
        sys.exit(0)
    except FileNotFoundError:
        print("ERROR: 'jupyter' command not found. Please install JupyterLab:")
        print("  pip install jupyterlab")
        sys.exit(1)


if __name__ == '__main__':
    main()
