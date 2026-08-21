"""Memory abstractions for AugAgent."""

from pydantic import BaseModel, Field
from typing import Any, List
import chromadb

class ShortTermMemory(BaseModel):
    """In-memory sliding window of recent task context."""
    recent_contexts: List[str] = Field(default_factory=list)
    max_items: int = 5
    
    def add_context(self, context: str):
        self.recent_contexts.append(context)
        if len(self.recent_contexts) > self.max_items:
            self.recent_contexts.pop(0)

class LongTermMemory:
    """Vector database abstraction using ChromaDB for persistent storage and RAG."""
    def __init__(self, collection_name: str = "augagent_kb", persist_directory: str = "./.chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
    def add_document(self, text: str, metadata: dict[str, Any] = None, doc_id: str = None):
        if not doc_id:
            import uuid
            doc_id = str(uuid.uuid4())
            
        self.collection.upsert(
            documents=[text],
            metadatas=[metadata or {}],
            ids=[doc_id]
        )
        
    def search(self, query: str, top_k: int = 3) -> List[dict[str, Any]]:
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        docs = []
        if results and results["documents"] and len(results["documents"]) > 0:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                docs.append({"text": doc, "metadata": meta})
                
        return docs

# Global instance for demonstration purposes
global_long_term_memory = LongTermMemory()
