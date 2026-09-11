import pytest
from src.preprocessing import clean_text, tokenize_text

def test_tokenize_empty_and_normal():
    assert tokenize_text("") == []
    assert tokenize_text(None) == []
    tokens = tokenize_text("Hello World! This is a test.")
    assert "hello" in tokens
    assert "world" in tokens

def test_tech_terms_preservation():
    text = "We need experience in C++, C#, Node.js, .NET, and SQL."
    tokens = tokenize_text(text)
    assert "c++" in tokens
    assert "c#" in tokens
    assert "node.js" in tokens
    assert ".net" in tokens
    assert "sql" in tokens

def test_clean_text():
    text = "Contact me at john@example.com or https://example.com. Working on Python and React!"
    cleaned = clean_text(text)
    assert "john@example.com" not in cleaned
    assert "https://example.com" not in cleaned
    assert "python" in cleaned
    assert "react" in cleaned
