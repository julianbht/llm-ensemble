#!/bin/bash
set -e

# Update system
sudo apt update && sudo apt upgrade -y

# Install python
sudo apt install python3 python3-pip python3-venv

# nvm to get latest npm for claude
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
nvm install --lts

# claude code
npm install -g @anthropic-ai/claude-code

