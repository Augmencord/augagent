import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# Basic PII redaction patterns
PII_PATTERNS = {
    "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    "credit_card": re.compile(r'\b(?:\d[ -]*?){13,16}\b'),
    "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
}

import hashlib
import time
class AuditLogger:
    """
    Writes immutable JSONL audit logs with basic PII redaction and retention policies.
    """
    def __init__(self, log_dir: str = ".audit_logs", max_retention_days: int = 30):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_retention_days = max_retention_days
        self.log_file = self.log_dir / f"audit_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"
        self._pii_patterns = {
            "email": re.compile(r"[\w\.-]+@[\w\.-]+\.\w+"),
            "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
            "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
        }
        self._sensitive_keys = {"api_key", "jwt_secret", "password", "token", "secret"}

    def _redact_pii(self, text: str) -> str:
        if not isinstance(text, str):
            return text
        for pii_type, pattern in self._pii_patterns.items():
            text = pattern.sub(f"[REDACTED_{pii_type.upper()}]", text)
        return text

    def _redact_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively redact PII and sensitive fields in dictionaries."""
        redacted: Dict[str, Any] = {}
        for k, v in data.items():
            if k.lower() in self._sensitive_keys:
                redacted[k] = "[REDACTED_SECRET]"
            elif isinstance(v, str):
                redacted[k] = self._redact_pii(v)
            elif isinstance(v, dict):
                redacted[k] = self._redact_dict(v)
            elif isinstance(v, list):
                redacted[k] = [
                    self._redact_dict(i) if isinstance(i, dict) 
                    else self._redact_pii(i) if isinstance(i, str) else i 
                    for i in v
                ]
            else:
                redacted[k] = v
        return redacted

    def log_event(self, event_type: str, data: Dict[str, Any], tenant_id: str = "default"):
        """Log an event to the audit file with SHA-256 checksum."""
        try:
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tenant_id": tenant_id,
                "event_type": event_type,
                "data": self._redact_dict(data)
            }
            
            entry_json = json.dumps(entry, sort_keys=True)
            checksum = hashlib.sha256(entry_json.encode('utf-8')).hexdigest()
            entry["checksum"] = checksum
            
            # Append only (immutable log simulation)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logging.error(f"Failed to write audit log: {e}")

    def log_tool_execution(self, tool_name: str, args: Dict[str, Any], result: Any, duration: float, agent_id: str, tenant_id: str = "default"):
        """Convenience wrapper for logging tool executions."""
        self.log_event(
            event_type="tool_execution",
            data={
                "tool_name": tool_name,
                "args": args,
                "result": str(result),
                "duration": duration,
                "agent_id": agent_id
            },
            tenant_id=tenant_id
        )

    def log_llm_call(self, model: str, prompt: str, response: str, tokens: Dict[str, int], latency: float, tenant_id: str = "default"):
        """Convenience wrapper for logging LLM calls."""
        self.log_event(
            event_type="llm_call",
            data={
                "model": model,
                "prompt": prompt,
                "response": response,
                "tokens": tokens,
                "latency": latency
            },
            tenant_id=tenant_id
        )

    def cleanup(self):
        """Deletes log files older than max_retention_days."""
        try:
            now = time.time()
            cutoff = now - (self.max_retention_days * 86400)
            
            for log_path in self.log_dir.glob("audit_*.jsonl"):
                if log_path.stat().st_mtime < cutoff:
                    log_path.unlink()
        except Exception as e:
            logging.error(f"Failed to cleanup audit logs: {e}")

# Global audit logger instance
audit = AuditLogger()

