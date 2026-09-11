import pytest
from src.experience_extractor import extract_experience, experience_match

def test_extract_experience_patterns():
    assert extract_experience("I have 2 years of experience.") == 2.0
    assert extract_experience("Minimum 3+ years in backend.") == 3.0
    assert extract_experience("Worked for 1.5 yrs.") == 1.5
    assert extract_experience("Experience: 4 years") == 4.0
    assert extract_experience("No experience stated.") == 0.0

def test_experience_match():
    assert experience_match(3.0, 2.0) == 1.0
    assert experience_match(2.0, 2.0) == 1.0
    assert experience_match(1.0, 2.0) == 0.5
    assert experience_match(0.0, 3.0) == 0.0
    assert experience_match(5.0, 0.0) == 1.0
