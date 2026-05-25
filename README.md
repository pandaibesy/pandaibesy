# Pandaibesy

Offline persistent memory CLI for Claude Code and Aider.
Solves AI amnesia across sessions — no cloud, no pip install.

## Setup

git clone https://github.com/pandaibesy/pandaibesy.git
cd pandaibesy
python nalar_bridge.py

## Usage

python nalar_bridge.py capture "your decision" project
python nalar_bridge.py query "search term"
python nalar_bridge.py mcp-pull "active context"
python nalar_bridge.py stats

## Why

Every Claude Code session forgets everything.
Pandaibesy stores your decisions locally in SQLite
and injects them back next session via MCP.

100% offline. Zero dependencies. Works on Termux.
