import abc
import json
from typing import Any, Dict, List
from datetime import datetime, timezone

try:
    import aiosqlite  # type: ignore
except ImportError:
    aiosqlite = None  # type: ignore

class BaseCheckpointer(abc.ABC):
    """Abstract interface for persisting and loading agent state."""

    @abc.abstractmethod
    async def save(self, thread_id: str, state: Dict[str, Any], step_index: int) -> None:
        pass

    @abc.abstractmethod
    async def load(self, thread_id: str) -> Dict[str, Any] | None:
        pass

    @abc.abstractmethod
    async def list(self, thread_id: str) -> List[Dict[str, Any]]:
        pass
        
    @abc.abstractmethod
    async def delete(self, thread_id: str) -> None:
        pass


class MemoryCheckpointer(BaseCheckpointer):
    """In-memory checkpointer for testing."""
    def __init__(self):
        # thread_id -> list of (step_index, state)
        self._store: Dict[str, List[tuple[int, Dict[str, Any]]]] = {}

    async def save(self, thread_id: str, state: Dict[str, Any], step_index: int) -> None:
        if thread_id not in self._store:
            self._store[thread_id] = []
        self._store[thread_id].append((step_index, state))
        # Keep ordered by step_index
        self._store[thread_id].sort(key=lambda x: x[0])

    async def load(self, thread_id: str) -> Dict[str, Any] | None:
        checkpoints = self._store.get(thread_id, [])
        if not checkpoints:
            return None
        return checkpoints[-1][1]

    async def list(self, thread_id: str) -> List[Dict[str, Any]]:
        checkpoints = self._store.get(thread_id, [])
        return [state for _, state in checkpoints]

    async def delete(self, thread_id: str) -> None:
        if thread_id in self._store:
            del self._store[thread_id]


class SQLiteCheckpointer(BaseCheckpointer):
    """SQLite-backed checkpointer for persistent state."""
    
    def __init__(self, db_path: str = ".checkpoints.sqlite"):
        self.db_path = db_path

    async def setup(self):
        if aiosqlite is None:
            raise ImportError("aiosqlite is required for SQLiteCheckpointer. Install with `pip install aiosqlite`")
        async with aiosqlite.connect(self.db_path) as db:  # type: ignore
            await db.execute('''
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    step_index INTEGER NOT NULL
                )
            ''')
            await db.commit()

    async def save(self, thread_id: str, state: Dict[str, Any], step_index: int) -> None:
        await self.setup()
        async with aiosqlite.connect(self.db_path) as db:  # type: ignore
            await db.execute('''
                INSERT INTO checkpoints (thread_id, state_json, created_at, step_index)
                VALUES (?, ?, ?, ?)
            ''', (thread_id, json.dumps(state), datetime.now(timezone.utc).isoformat(), step_index))
            await db.commit()

    async def load(self, thread_id: str) -> Dict[str, Any] | None:
        await self.setup()
        async with aiosqlite.connect(self.db_path) as db:  # type: ignore
            async with db.execute('''
                SELECT state_json FROM checkpoints 
                WHERE thread_id = ? 
                ORDER BY step_index DESC LIMIT 1
            ''', (thread_id,)) as cursor:  # type: ignore
                row = await cursor.fetchone()
                if row:
                    return json.loads(str(row[0]))  # type: ignore
                return None

    async def list(self, thread_id: str) -> List[Dict[str, Any]]:
        await self.setup()
        async with aiosqlite.connect(self.db_path) as db:  # type: ignore
            async with db.execute('''
                SELECT state_json FROM checkpoints 
                WHERE thread_id = ? 
                ORDER BY step_index ASC
            ''', (thread_id,)) as cursor:  # type: ignore
                rows = await cursor.fetchall()
                return [json.loads(str(row[0])) for row in rows]  # type: ignore
                
    async def delete(self, thread_id: str) -> None:
        await self.setup()
        async with aiosqlite.connect(self.db_path) as db:  # type: ignore
            await db.execute('DELETE FROM checkpoints WHERE thread_id = ?', (thread_id,))
            await db.commit()
