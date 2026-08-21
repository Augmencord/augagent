import pytest
from unittest.mock import patch, MagicMock
from augagent.checkpoint import SQLiteCheckpointer, MemoryCheckpointer

def test_memory_checkpointer():
    cp = MemoryCheckpointer()
    cp.save("agent-123", {"foo": "bar"})
    assert cp.load("agent-123") == {"foo": "bar"}
    assert cp.load("agent-456") is None

def test_sqlite_checkpointer(tmp_path):
    db_path = tmp_path / "test_checkpoints.db"
    cp = SQLiteCheckpointer(str(db_path))
    
    # Test saving
    state = {"history": [{"role": "user", "content": "hello"}]}
    cp.save("agent-123", state)
    
    # Test loading
    loaded = cp.load("agent-123")
    assert loaded == state
    
    # Test update existing
    state["history"].append({"role": "assistant", "content": "hi"})
    cp.save("agent-123", state)
    
    loaded2 = cp.load("agent-123")
    assert loaded2 == state
    
    # Test missing
    assert cp.load("missing-agent") is None
