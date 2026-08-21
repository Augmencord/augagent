import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Basic PII redaction patterns
PII_PATTERNS = {
    "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    "credit_card": re.compile(r'\b(?:\d[ -]*?){13,16}\b'),
    "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
}

class AuditLogger:
    """
    Writes immutable JSONL audit logs with basic PII redaction.
    """
    def __init__(self, log_dir: str = "audit_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # Use a daily log file
        self.log_file = self.log_dir / f"audit_{datetime.utcnow().strftime('%Y-%m-%d')}.jsonl"

    def _redact_pii(self, text: str) -> str:
        if not isinstance(text, str):
            return text
        for pii_type, pattern in PII_PATTERNS.items():
            text = pattern.sub(f"[REDACTED_{pii_type.upper()}]", text)
        return text

    def _redact_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        redacted = {}
        for k, v in data.items():
            if isinstance(v, str):
                redacted[k] = self._redact_pii(v)
            elif isinstance(v, dict):
                redacted[k] = self._redact_dict(v)
            elif isinstance(v, list):
                redacted[k] = [self._redact_pii(i) if isinstance(i, str) else i for i in v]
            else:
                redacted[k] = v
        return redacted

    def log_event(self, event_type: str, actor: str, payload: Dict[str, Any]):
        """Log an event to the audit file."""
        try:
            entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "event_type": event_type,
                "actor": actor,
                "payload": self._redact_dict(payload)
            }
            # Append only (immutable log simulation)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logging.error(f"Failed to write audit log: {e}")

# Global audit logger instance
audit = AuditLogger()
