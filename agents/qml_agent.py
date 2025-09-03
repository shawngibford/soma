from __future__ import annotations

import yaml
from typing import Any, Dict

from agents.llm_ollama import OllamaClient
from agents.base_agent import BaseAgent


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_qml_agent(config_path: str) -> BaseAgent:
    cfg = load_config(config_path)

    llm = OllamaClient()

    return BaseAgent(
        name=cfg["name"],
        system_prompt=cfg["system_prompt"],
        llm=llm,
        model_name=cfg["model"]["name"],
        persistence=cfg["persistence"],
        runtime={
            "stream": bool(cfg["runtime"].get("stream", True)),
            "temperature": float(cfg["model"].get("temperature", 0.2)),
            "top_p": float(cfg["model"].get("top_p", 0.95)),
            "max_tokens": int(cfg["model"].get("max_tokens", 1024)),
            "timeout_seconds": int(cfg["runtime"].get("timeout_seconds", 60)),
        },
    )
