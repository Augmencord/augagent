import pytest
import os
import json
from augagent.audit import AuditLogger

def test_audit_pii_redaction(tmp_path):
    logger = AuditLogger(log_dir=str(tmp_path))
    
    # Test strings with PII
    email_text = "Contact me at user@example.com for more info."
    card_text = "My card is 1234-5678-9012-3456."
    ssn_text = "Social security: 123-45-6789."
    
    assert logger._redact_pii(email_text) == "Contact me at [REDACTED_EMAIL] for more info."
    assert logger._redact_pii(card_text) == "My card is [REDACTED_CREDIT_CARD]."
    assert logger._redact_pii(ssn_text) == "Social security: [REDACTED_SSN]."

def test_audit_dict_redaction(tmp_path):
    logger = AuditLogger(log_dir=str(tmp_path))
    
    payload = {
        "user_email": "test@example.com",
        "nested": {
            "ssn": "987-65-4321",
            "safe_val": 42
        },
        "list_data": ["normal string", "card 1111-2222-3333-4444"]
    }
    
    redacted = logger._redact_dict(payload)
    
    assert redacted["user_email"] == "[REDACTED_EMAIL]"
    assert redacted["nested"]["ssn"] == "[REDACTED_SSN]"
    assert redacted["nested"]["safe_val"] == 42
    assert redacted["list_data"][1] == "card [REDACTED_CREDIT_CARD]"

def test_audit_log_event(tmp_path):
    logger = AuditLogger(log_dir=str(tmp_path))
    
    payload = {"msg": "Hello user@test.com"}
    logger.log_event("TEST_EVENT", "test_actor", payload)
    
    # Verify file was created and contains redacted data
    assert logger.log_file.exists()
    
    with open(logger.log_file, "r") as f:
        line = f.readline()
        data = json.loads(line)
        
        assert data["event_type"] == "TEST_EVENT"
        assert data["actor"] == "test_actor"
        assert data["payload"]["msg"] == "Hello [REDACTED_EMAIL]"
