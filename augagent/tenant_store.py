"""Durable tenant store for AugAgent multi-tenancy."""

import sqlite3
from typing import Any, Dict, List, Optional
import datetime

try:
    import aiosqlite
    HAS_AIOSQLITE = True
except ImportError:
    HAS_AIOSQLITE = False

class SQLiteTenantStore:
    def __init__(self, db_path: str = ".tenants.sqlite"):
        if not HAS_AIOSQLITE:
            raise ImportError("aiosqlite is required. Install with `pip install aiosqlite`")
        self.db_path = db_path

    async def setup(self):
        async with aiosqlite.connect(self.db_path) as db:  # type: ignore
            await db.execute('''
                CREATE TABLE IF NOT EXISTS tenants (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    api_calls INTEGER DEFAULT 0,
                    tokens INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                )
            ''')
            # Insert default tenant if not exists
            await db.execute('''
                INSERT OR IGNORE INTO tenants (id, name, status, api_calls, tokens, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', ("default", "Default Tenant", "active", 0, 0, datetime.datetime.now(datetime.timezone.utc).isoformat()))
            await db.commit()

    async def create_tenant(self, tenant_id: str, name: str) -> bool:
        await self.setup()
        async with aiosqlite.connect(self.db_path) as db:  # type: ignore
            try:
                now = datetime.datetime.now(datetime.timezone.utc).isoformat()
                await db.execute('''
                    INSERT INTO tenants (id, name, status, api_calls, tokens, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (tenant_id, name, "active", 0, 0, now))
                await db.commit()
                return True
            except aiosqlite.IntegrityError:
                return False

    async def list_tenants(self) -> List[Dict[str, Any]]:
        await self.setup()
        async with aiosqlite.connect(self.db_path) as db:  # type: ignore
            async with db.execute('SELECT * FROM tenants') as cursor:
                rows = await cursor.fetchall()
                results = []
                for row in rows:
                    columns = [description[0] for description in cursor.description]
                    results.append(dict(zip(columns, row)))
                return results

    async def get_tenant(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        await self.setup()
        async with aiosqlite.connect(self.db_path) as db:  # type: ignore
            async with db.execute('SELECT * FROM tenants WHERE id = ?', (tenant_id,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    columns = [description[0] for description in cursor.description]
                    return dict(zip(columns, row))
                return None

    async def deactivate_tenant(self, tenant_id: str) -> bool:
        await self.setup()
        async with aiosqlite.connect(self.db_path) as db:  # type: ignore
            cursor = await db.execute('UPDATE tenants SET status = ? WHERE id = ?', ("inactive", tenant_id))
            await db.commit()
            return cursor.rowcount > 0

tenant_store = SQLiteTenantStore()
