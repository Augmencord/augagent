import pytest
from augagent.tools import aug_tool, AugTool
from typing import Literal

def test_tool_schema_generation():
    @aug_tool
    def complex_tool(
        name: str,
        age: int,
        is_active: bool = True,
        role: Literal["admin", "user"] = "user"
    ) -> str:
        """Create a user.
        
        Args:
            name: The user's name
            age: The user's age
            is_active: Whether active
            role: The role
        """
        return "success"
        
    schema = complex_tool.to_openai_schema()
    
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "complex_tool"
    assert schema["function"]["description"].startswith("Create a user.")
    
    props = schema["function"]["parameters"]["properties"]
    assert "name" in props
    assert props["name"]["type"] == "string"
    
    assert "age" in props
    assert props["age"]["type"] == "integer"
    
    assert "is_active" in props
    assert props["is_active"]["type"] == "boolean"
    
    assert "role" in props
    assert props["role"]["type"] == "string"
    assert "enum" in props["role"]
    assert props["role"]["enum"] == ["admin", "user"]
    
    required = schema["function"]["parameters"]["required"]
    assert "name" in required
    assert "age" in required
    assert "is_active" not in required
    assert "role" not in required

def test_from_function():
    def standard_func(x: int) -> int:
        """Multiply by 2."""
        return x * 2
        
    wrapped = AugTool.from_function(standard_func)
    assert wrapped.name == "standard_func"
    assert wrapped.to_openai_schema()["function"]["description"] == "Multiply by 2."
    
    result = wrapped.func(x=5)
    assert result == 10
