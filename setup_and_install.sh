#!/bin/bash
# Setup and Installation Script for DeepAgent Lab Extension

set -e  # Exit on error

echo "🚀 DeepAgent Lab - Installation Script"
echo "============================================"
echo ""

# Check if virtual environment exists
VENV_PATH="/Users/dkedar7/.venvs/deepagent-lab"

if [ ! -d "$VENV_PATH" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv "$VENV_PATH"
fi

echo "✅ Activating virtual environment..."
source "$VENV_PATH/bin/activate"

echo "📦 Installing Python dependencies..."
uv pip install --upgrade pip
uv pip install jupyter-server jupyterlab langgraph langchain-core langchain python-dotenv

echo "📦 Installing JavaScript dependencies..."
yarn install

echo "🔨 Building TypeScript code..."
yarn build:lib

echo "🔨 Building JupyterLab extension..."
jupyter labextension build .

echo "📦 Installing Python package..."
uv pip install -e .

echo "✅ Enabling server extension..."
jupyter server extension enable deepagent_lab

echo "🔗 Linking labextension..."
jupyter labextension develop . --overwrite

echo ""
echo "✅ Installation complete!"
echo ""
echo "📋 Verification:"
echo "----------------"
jupyter labextension list | grep deepagents
jupyter server extension list | grep deepagents

echo ""
echo "🎉 All done! To start JupyterLab, run:"
echo "   source $VENV_PATH/bin/activate"
echo "   jupyter lab"
