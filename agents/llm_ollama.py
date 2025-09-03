import os
import json
from typing import Iterable, Dict, Any, Optional, Union

import httpx


class OllamaClient:
    """Minimal client for the Ollama HTTP API (local by default).

    Supports streaming chat responses and non-streaming calls.
    """

    def __init__(self, host: Optional[str] = None) -> None:
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self._chat_url = f"{self.host}/api/chat"

    def chat(
        self,
        model: str,
        messages: list,
        stream: bool = True,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 1024,
        timeout: int = 60,
        **_: Any,
    ) -> Union[Iterable[str], Dict[str, Any]]:
        """Call the Ollama chat API.

        If stream=True, yields text deltas (str). Otherwise returns the JSON response (dict).
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            # Ollama generation controls live under "options"
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        }

        if stream:
            def _gen() -> Iterable[str]:
                with httpx.Client(timeout=timeout) as client:
                    with client.stream("POST", self._chat_url, json=payload) as r:
                        r.raise_for_status()
                        for raw in r.iter_lines():
                            if not raw:
                                continue
                            line = raw.strip()  # raw is already a string from iter_lines()
                            # Some servers prefix with "data: "; strip if present
                            if line.startswith("data: "):
                                line = line[6:].strip()
                            try:
                                obj = json.loads(line)
                            except Exception:
                                # Keep-alives or non-JSON chunks
                                continue
                            # Chat streaming chunks typically have message.content deltas
                            delta = ""
                            msg = obj.get("message") or {}
                            if isinstance(msg, dict):
                                delta = msg.get("content", "")
                            if not delta:
                                # Fallback for other schemas
                                delta = obj.get("response", "")
                            if delta:
                                yield delta
                            if obj.get("done") is True:
                                break
            return _gen()
        else:
            with httpx.Client(timeout=timeout) as client:
                r = client.post(self._chat_url, json=payload)
                r.raise_for_status()
                return r.json()

