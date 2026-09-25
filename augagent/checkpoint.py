"""State Checkpointing for AugAgent.

Provides dual sync/async persistence backends for agent session state.
"""

from __future__ import annotations

import abc
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class DualValue:
    """A wrapper that behaves as its inner value synchronously and is also awaitable."""

    def __init__(self, val: Any = None) -> None:
        self.val = val

    def __await__(self):
        async def _coro():
            return self.val
        return _coro().__await__()

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, DualValue):
            return self.val == other.val
        return self.val == other

    def __bool__(self) -> bool:
        return bool(self.val)

    def __getitem__(self, key: Any) -> Any:
        return self.val[key]

    def __repr__(self) -> str:
        return repr(self.val)


class BaseCheckpointer(abc.ABC):
    """Abstract interface for persisting and loading agent state."""

    @abc.abstractmethod
    def save(self, thread_id: str, state: Dict[str, Any], step_index: int = 0) -> Any:
        pass

    @abc.abstractmethod
    def load(self, thread_id: str) -> Any:
        pass

    @abc.abstractmethod
    def list(self, thread_id: str) -> Any:
        pass

    @abc.abstractmethod
    def delete(self, thread_id: str) -> Any:
        pass


class MemoryCheckpointer(BaseCheckpointer):
    """In-memory checkpointer supporting both sync and async usage."""

    def __init__(self) -> None:
        self._store: Dict[str, List[tuple[int, Dict[str, Any]]]] = {}

    def save(self, thread_id: str, state: Dict[str, Any], step_index: int = 0) -> Any:
        if thread_id not in self._store:
            self._store[thread_id] = []
        self._store[thread_id].append((step_index, state))
        self._store[thread_id].sort(key=lambda x: x[0])
        return DualValue(None)

    def load(self, thread_id: str) -> Any:
        checkpoints = self._store.get(thread_id, [])
        if not checkpoints:
            return None
        return DualValue(checkpoints[-1][1])

    def list(self, thread_id: str) -> Any:
        checkpoints = self._store.get(thread_id, [])
        return DualValue([state for _, state in checkpoints])

    def delete(self, thread_id: str) -> Any:
        if thread_id in self._store:
            del self._store[thread_id]
        return DualValue(None)


class SQLiteCheckpointer(BaseCheckpointer):
    """SQLite-backed checkpointer supporting both sync and async usage."""

    def __init__(self, db_path: str = ".checkpoints.sqlite") -> None:
        self.db_path = db_path
        self._setup()

    def _setup(self) -> None:
        with sqlite3.connect(self.db_path) as db:
            db.execute('''
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    step_index INTEGER NOT NULL
                )
            ''')
            db.commit()

    def save(self, thread_id: str, state: Dict[str, Any], step_index: int = 0) -> Any:
        self._setup()
        with sqlite3.connect(self.db_path) as db:
            db.execute('''
                INSERT INTO checkpoints (thread_id, state_json, created_at, step_index)
                VALUES (?, ?, ?, ?)
            ''', (thread_id, json.dumps(state), datetime.now(timezone.utc).isoformat(), step_index))
            db.commit()
        return DualValue(None)

    def load(self, thread_id: str) -> Any:
        self._setup()
        with sqlite3.connect(self.db_path) as db:
            cursor = db.execute('''
                SELECT state_json FROM checkpoints 
                WHERE thread_id = ? 
                ORDER BY step_index DESC, id DESC LIMIT 1
            ''', (thread_id,))
            row = cursor.fetchone()
            if row:
                return DualValue(json.loads(str(row[0])))
            return None

    def list(self, thread_id: str) -> Any:
        self._setup()
        with sqlite3.connect(self.db_path) as db:
            cursor = db.execute('''
                SELECT state_json FROM checkpoints 
                WHERE thread_id = ? 
                ORDER BY step_index ASC
            ''', (thread_id,))
            rows = cursor.fetchall()
            return DualValue([json.loads(str(row[0])) for row in rows])

    def delete(self, thread_id: str) -> Any:
        self._setup()
        with sqlite3.connect(self.db_path) as db:
            db.execute('DELETE FROM checkpoints WHERE thread_id = ?', (thread_id,))
            db.commit()
        return DualValue(None)
