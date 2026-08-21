import abc
import json
import sqlite3
from typing import Any, Dict

class BaseCheckpointer(abc.ABC):
    """Abstract interface for persisting and loading agent state."""

    @abc.abstractmethod
    def save(self, agent_id: str, state: Dict[str, Any]) -> None:
        """Save the agent's state."""
        pass

    @abc.abstractmethod
    def load(self, agent_id: str) -> Dict[str, Any] | None:
        """Load the agent's state. Returns None if no checkpoint exists."""
        pass


class MemoryCheckpointer(BaseCheckpointer):
    """In-memory checkpointer for testing."""
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    def save(self, agent_id: str, state: Dict[str, Any]) -> None:
        self._store[agent_id] = state

    def load(self, agent_id: str) -> Dict[str, Any] | None:
        return self._store.get(agent_id)


class SQLiteCheckpointer(BaseCheckpointer):
    """SQLite-backed checkpointer for persistent state."""
    
    def __init__(self, db_path: str = ".checkpoints.sqlite"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS checkpoints (
                    agent_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL
                )
            ''')
            conn.commit()

    def save(self, agent_id: str, state: Dict[str, Any]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO checkpoints (agent_id, state)
                VALUES (?, ?)
                ON CONFLICT(agent_id) DO UPDATE SET state=excluded.state
            ''', (agent_id, json.dumps(state)))
            conn.commit()

    def load(self, agent_id: str) -> Dict[str, Any] | None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT state FROM checkpoints WHERE agent_id = ?', (agent_id,))
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
            return None

# EOF
