#!/usr/bin/env python3
import argparse
import os
import sys

# Ensure the project root is on sys.path so local 'agents' is imported, not site-packages
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.qml_agent import build_qml_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Soma Senior QML Agent CLI")
    parser.add_argument("--config", default="config/qml.yaml", help="Path to QML agent config")
    args = parser.parse_args()

    # Resolve config relative to project root when a relative path is provided
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)
    if not os.path.exists(config_path):
        parser.error(f"Config file not found: {config_path}")

    agent = build_qml_agent(config_path)
    print(f"Loaded QML agent: {agent.name}")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user = input("QML> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if user.lower() in {"exit", "quit"}:
            break
        agent.chat(user)


if __name__ == "__main__":
    main()
