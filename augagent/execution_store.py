"""Durable execution store for AugAgent."""

import abc
import json
import sqlite3
import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    import aiosqlite
    HAS_AIOSQLITE = True
except ImportError:
    HAS_AIOSQLITE = False

class ExecutionStatus(str, Enum):
    RUNNING = "RUNNING"
    WAITING_HUMAN_INPUT = "WAITING_HUMAN_INPUT"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class BaseExecutionStore(abc.ABC):
    """Abstract interface for persisting execution state across restarts."""

    @abc.abstractmethod
    async def save_state(self, thread_id: str, agent_config: Dict[str, Any], message_history: List[Dict[str, Any]], current_step: int, status: ExecutionStatus, pending_action: Optional[Dict[str, Any]] = None) -> None:
        pass

    @abc.abstractmethod
    async def load_state(self, thread_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    async def update_status(self, thread_id: str, status: ExecutionStatus) -> None:
        pass

    @abc.abstractmethod
    async def get_pending_executions(self) -> List[Dict[str, Any]]:
        pass


class SQLiteExecutionStore(BaseExecutionStore):
    """SQLite implementation of durable execution store."""

    def __init__(self, db_path: str = ".executions.sqlite"):
        if not HAS_AIOSQLITE:
            raise ImportError("aiosqlite is required. Install with `pip install aiosqlite`")
        self.db_path = db_path

    async def setup(self):
        async with aiosqlite.connect(self.db_path) as db:  # type: ignore
            await db.execute('''
                CREATE TABLE IF NOT EXISTS execution_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT UNIQUE NOT NULL,
                    agent_config_json TEXT NOT NULL,
                    message_history_json TEXT NOT NULL,
                    current_step INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    pending_action_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            await db.commit()

    async def save_state(self, thread_id: str, agent_config: Dict[str, Any], message_history: List[Dict[str, Any]], current_step: int, status: ExecutionStatus, pending_action: Optional[Dict[str, Any]] = None) -> None:
        await self.setup()
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        async with aiosqlite.connect(self.db_path) as db:  # type: ignore
            await db.execute('''
                INSERT INTO execution_states (thread_id, agent_config_json, message_history_json, current_step, status, pending_action_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    agent_config_json = excluded.agent_config_json,
                    message_history_json = excluded.message_history_json,
                    current_step = excluded.current_step,
                    status = excluded.status,
                    pending_action_json = excluded.pending_action_json,
                    updated_at = excluded.updated_at
            ''', (
                thread_id,
                json.dumps(agent_config),
                json.dumps(message_history),
                current_step,
                status.value,
                json.dumps(pending_action) if pending_action else None,
                now,
                now
            ))
            await db.commit()

    async def load_state(self, thread_id: str) -> Optional[Dict[str, Any]]:
        await self.setup()
        async with aiosqlite.connect(self.db_path) as db:  # type: ignore
            async with db.execute('SELECT * FROM execution_states WHERE thread_id = ?', (thread_id,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    columns = [description[0] for description in cursor.description]
                    data = dict(zip(columns, row))
                    
                    data['agent_config'] = json.loads(data['agent_config_json'])
                    data['message_history'] = json.loads(data['message_history_json'])
                    data['pending_action'] = json.loads(data['pending_action_json']) if data['pending_action_json'] else None
                    data['status'] = ExecutionStatus(data['status'])
                    return data
                return None

    async def update_status(self, thread_id: str, status: ExecutionStatus) -> None:
        await self.setup()
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        async with aiosqlite.connect(self.db_path) as db:  # type: ignore
            await db.execute('UPDATE execution_states SET status = ?, updated_at = ? WHERE thread_id = ?', (status.value, now, thread_id))
            await db.commit()

    async def get_pending_executions(self) -> List[Dict[str, Any]]:
        await self.setup()
        async with aiosqlite.connect(self.db_path) as db:  # type: ignore
            async with db.execute('SELECT thread_id, agent_config_json, status, current_step FROM execution_states WHERE status = ?', (ExecutionStatus.WAITING_HUMAN_INPUT.value,)) as cursor:
                rows = await cursor.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "thread_id": row[0],
                        "agent_config": json.loads(row[1]),
                        "status": row[2],
                        "current_step": row[3]
                    })
                return results

execution_store = SQLiteExecutionStore()
