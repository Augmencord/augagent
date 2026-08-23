"""JSONL append-only checkpointer for agent state."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List
from augagent.checkpoint import BaseCheckpointer

class JsonlCheckpointer(BaseCheckpointer):
    """Stores agent checkpoints as append-only JSONL files, highly efficient for large contexts."""
    
    def __init__(self, base_dir: str = ".checkpoints"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, thread_id: str) -> Path:
        return self.base_dir / f"{thread_id}.jsonl"

    async def save(self, thread_id: str, state: Dict[str, Any], step_index: int) -> None:
        file_path = self._get_file_path(thread_id)
        
        record = {
            "thread_id": thread_id,
            "step_index": step_index,
            "state": state
        }
        
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    async def load(self, thread_id: str) -> Dict[str, Any] | None:
        file_path = self._get_file_path(thread_id)
        if not file_path.exists():
            return None
            
        last_state = None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        last_state = record.get("state")
        except Exception:
            pass
            
        return last_state

    async def list(self, thread_id: str) -> List[Dict[str, Any]]:
        file_path = self._get_file_path(thread_id)
        if not file_path.exists():
            return []
            
        states = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        states.append(record.get("state", {}))
        except Exception:
            pass
            
        return states

    async def delete(self, thread_id: str) -> None:
        file_path = self._get_file_path(thread_id)
        if file_path.exists():
            try:
                os.remove(file_path)
            except OSError:
                pass
