import json
from pathlib import Path
from typing import Dict, Any, List, Optional


class BaseAgent:
    """Base agent with transcript persistence and simple chat interface."""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        llm,
        model_name: str,
        persistence: Dict[str, str],
        runtime: Dict[str, Any],
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm
        self.model_name = model_name
        self.runtime = runtime
        self.paths = {k: Path(v) for k, v in persistence.items()}
        for p in self.paths.values():
            p.parent.mkdir(parents=True, exist_ok=True)

    # ----- Persistence helpers -----
    def _append_jsonl(self, path: Path, obj: Dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _load_transcript(self) -> List[Dict[str, str]]:
        path = self.paths.get("transcript_path")
        if not path or not path.exists():
            return []
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    # ----- Core chat -----
    def chat(self, user_message: str, meta: Optional[Dict[str, Any]] = None) -> str:
        """Send a message and stream/return the assistant response.

        Persists the conversation to transcript JSONL.
        """
        # Build context (limited rolling window)
        messages: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        for m in self._load_transcript()[-15:]:
            # Expecting {"role": "user"|"assistant", "content": str}
            if m.get("role") in {"user", "assistant"}:
                messages.append(m)
        messages.append({"role": "user", "content": user_message})

        # Generation controls
        stream = bool(self.runtime.get("stream", True))
        gen_kwargs = {
            "temperature": float(self.runtime.get("temperature", 0.2)),
            "top_p": float(self.runtime.get("top_p", 0.95)),
            "max_tokens": int(self.runtime.get("max_tokens", 1024)),
            "timeout": int(self.runtime.get("timeout_seconds", 60)),
        }

        if stream:
            response_text = ""
            for delta in self.llm.chat(
                model=self.model_name,
                messages=messages,
                stream=True,
                **gen_kwargs,
            ):
                response_text += delta
                print(delta, end="", flush=True)
            print()
        else:
            obj = self.llm.chat(
                model=self.model_name,
                messages=messages,
                stream=False,
                **gen_kwargs,
            )
            # Try to read content from the response
            response_text = ""
            if isinstance(obj, dict):
                msg = obj.get("message") or {}
                if isinstance(msg, dict):
                    response_text = msg.get("content", "")
                if not response_text:
                    response_text = obj.get("response", "")

        # Persist the exchange
        self._append_jsonl(self.paths["transcript_path"], {"role": "user", "content": user_message})
        self._append_jsonl(self.paths["transcript_path"], {"role": "assistant", "content": response_text})
        if meta is not None and "decisions_path" in self.paths:
            self._append_jsonl(self.paths["decisions_path"], meta)
        return response_text

