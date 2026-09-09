from __future__ import annotations
import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ShortTermMemory:
    """Rolling conversation buffer for the *current* session only."""

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self.turns: List[Dict[str, Any]] = []

    def add(self, role: str, content: str, extra: Optional[Dict] = None) -> None:
        entry = {"role": role, "content": content, "ts": _now()}
        if extra:
            entry.update(extra)
        self.turns.append(entry)
        # keep only the last N turns to bound context size
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def as_list(self) -> List[Dict[str, Any]]:
        return list(self.turns)

    def clear(self) -> None:
        self.turns = []


class LongTermMemory:
    """
    Persistent, per-user memory backed by a JSON file on disk.
    Thread-safe for simple single-process use via a lock.

    Structure on disk:
    {
      "U100": {
          "facts": {"preferred_category": "Footwear", ...},
          "open_cases": [{"type": "return", "order_id": "ORD-10001", "status": "pending", ...}],
          "interaction_log": [{"ts": ..., "summary": "..."}]
      },
      ...
    }
    """

    def __init__(self, path: str = "data/long_term_memory.json"):
        self.path = path
        self._lock = threading.Lock()
        if not os.path.exists(self.path):
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self._write({})

    # ---------- low level ----------
    def _read(self) -> Dict[str, Any]:
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, data: Dict[str, Any]) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _ensure_user(self, data: Dict[str, Any], user_id: str) -> None:
        if user_id not in data:
            data[user_id] = {"facts": {}, "open_cases": [], "interaction_log": []}

    # ---------- public API ----------
    def get_profile(self, user_id: str) -> Dict[str, Any]:
        with self._lock:
            data = self._read()
            self._ensure_user(data, user_id)
            return data[user_id]

    def set_fact(self, user_id: str, key: str, value: Any) -> None:
        with self._lock:
            data = self._read()
            self._ensure_user(data, user_id)
            data[user_id]["facts"][key] = value
            self._write(data)

    def get_fact(self, user_id: str, key: str, default: Any = None) -> Any:
        return self.get_profile(user_id)["facts"].get(key, default)

    def add_open_case(self, user_id: str, case: Dict[str, Any]) -> None:
        with self._lock:
            data = self._read()
            self._ensure_user(data, user_id)
            case = dict(case)
            case.setdefault("created_at", _now())
            data[user_id]["open_cases"].append(case)
            self._write(data)

    def update_case_status(self, user_id: str, case_id: str, status: str) -> bool:
        with self._lock:
            data = self._read()
            self._ensure_user(data, user_id)
            updated = False
            for case in data[user_id]["open_cases"]:
                if case.get("case_id") == case_id:
                    case["status"] = status
                    case["updated_at"] = _now()
                    updated = True
            if updated:
                self._write(data)
            return updated

    def log_interaction(self, user_id: str, summary: str) -> None:
        with self._lock:
            data = self._read()
            self._ensure_user(data, user_id)
            data[user_id]["interaction_log"].append({"ts": _now(), "summary": summary})
            # bound the log so the file doesn't grow unbounded
            data[user_id]["interaction_log"] = data[user_id]["interaction_log"][-200:]
            self._write(data)

    def recent_interactions(self, user_id: str, n: int = 5) -> List[Dict[str, Any]]:
        return self.get_profile(user_id)["interaction_log"][-n:]
