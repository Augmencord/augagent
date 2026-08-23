"""Memory abstractions for AugAgent."""

import abc
import json
import sqlite3
import datetime
from pydantic import BaseModel, Field
from typing import Any, List, Optional, Dict

try:
    import chromadb
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

class BaseMemory(abc.ABC):
    """Abstract interface for long-term memory."""
    
    @abc.abstractmethod
    def store(self, text: str, metadata: dict[str, Any] | None = None, doc_id: str | None = None) -> None:
        pass
        
    @abc.abstractmethod
    def query(self, text: str, top_k: int = 3) -> List[dict[str, Any]]:
        pass
        
    @abc.abstractmethod
    def forget(self, doc_id: str) -> None:
        pass


class ShortTermMemory(BaseModel):
    """In-memory sliding window of recent task context."""
    recent_contexts: List[str] = Field(default_factory=list)
    max_items: int = 20
    
    def add_context(self, context: str):
        self.recent_contexts.append(context)
        if len(self.recent_contexts) > self.max_items:
            self.recent_contexts.pop(0)
            
    def get_context(self) -> str:
        return "\n".join(self.recent_contexts)


class ChromaMemory(BaseMemory):
    """Vector database abstraction using ChromaDB."""
    def __init__(self, collection_name: str = "augagent_kb", persist_directory: str = "./.chroma_db"):
        if not HAS_CHROMADB:
            raise ImportError("chromadb is not installed. Please install with `pip install augagent[memory]`")
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
    def store(self, text: str, metadata: dict[str, Any] | None = None, doc_id: str | None = None) -> None:
        if not doc_id:
            import uuid
            doc_id = str(uuid.uuid4())
            
        self.collection.upsert(
            documents=[text],
            metadatas=[metadata or {}],
            ids=[doc_id]
        )
        
    def query(self, text: str, top_k: int = 3) -> List[dict[str, Any]]:
        results = self.collection.query(
            query_texts=[text],
            n_results=top_k
        )
        
        docs = []
        docs_list = results.get("documents")
        if results and docs_list and len(docs_list) > 0:
            for i, doc in enumerate(docs_list[0]):
                meta_list = results.get("metadatas")
                meta = meta_list[0][i] if meta_list else {}
                docs.append({"text": doc, "metadata": meta})
                
        return docs
        
    def forget(self, doc_id: str) -> None:
        self.collection.delete(ids=[doc_id])


class QdrantMemory(BaseMemory):
    """Vector database abstraction using Qdrant."""
    def __init__(self, collection_name: str = "augagent_kb", location: str = ":memory:"):
        if not HAS_QDRANT:
            raise ImportError("qdrant-client is not installed.")
        self.client = QdrantClient(location=location)
        self.collection_name = collection_name
        
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            
    def store(self, text: str, metadata: dict[str, Any] | None = None, doc_id: str | None = None) -> None:
        import uuid
        if not doc_id: doc_id = str(uuid.uuid4())
        vector = [0.0] * 384
        payload = metadata or {}
        payload["text"] = text
        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=doc_id, vector=vector, payload=payload)]
        )

    def query(self, text: str, top_k: int = 3) -> List[dict[str, Any]]:
        vector = [0.0] * 384
        results = self.client.search(  # type: ignore
            collection_name=self.collection_name,
            query_vector=vector,
            limit=top_k
        )
        docs = []
        for res in results:
            docs.append({"text": res.payload.get("text", ""), "metadata": res.payload})
        return docs
        
    def forget(self, doc_id: str) -> None:
        self.client.delete(collection_name=self.collection_name, points_selector=[doc_id])


class InMemoryVectorStore(BaseMemory):
    """Simple in-memory store for testing without external deps."""
    def __init__(self):
        self.docs = {}
        
    def store(self, text: str, metadata: dict[str, Any] | None = None, doc_id: str | None = None) -> None:
        import uuid
        if not doc_id: doc_id = str(uuid.uuid4())
        self.docs[doc_id] = {"text": text, "metadata": metadata or {}}
        
    def query(self, text: str, top_k: int = 3) -> List[dict[str, Any]]:
        results = []
        for doc in self.docs.values():
            if text.lower() in doc["text"].lower():
                results.append(doc)
        return results[:top_k]
        
    def forget(self, doc_id: str) -> None:
        self.docs.pop(doc_id, None)


class EntityMemory:
    """Extracts and stores named entities in SQLite."""
    def __init__(self, db_path: str = ".entities.sqlite"):
        self.db_path = db_path
        self._setup()
        
    def _setup(self):
        with sqlite3.connect(self.db_path) as db:
            db.execute('''
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    type TEXT NOT NULL,
                    attributes_json TEXT NOT NULL,
                    last_seen TEXT NOT NULL
                )
            ''')
            db.commit()
            
    def store_entity(self, name: str, type: str, attributes: dict[str, Any]) -> None:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as db:
            db.execute('''
                INSERT INTO entities (name, type, attributes_json, last_seen)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    attributes_json = excluded.attributes_json,
                    last_seen = excluded.last_seen
            ''', (name, type, json.dumps(attributes), now))
            db.commit()
            
    def query_entity(self, name: str) -> Optional[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as db:
            cursor = db.execute('SELECT type, attributes_json, last_seen FROM entities WHERE name = ?', (name,))
            row = cursor.fetchone()
            if row:
                return {
                    "name": name,
                    "type": row[0],
                    "attributes": json.loads(row[1]),
                    "last_seen": row[2]
                }
            return None
