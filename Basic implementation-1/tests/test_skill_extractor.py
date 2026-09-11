import pytest
from src.skill_extractor import extract_skills, get_matched_skills, get_missing_skills

def test_extract_skills_case_insensitive_and_multiword():
    text = "Proficient in python, Machine Learning, FASTAPI, and Docker."
    skills = extract_skills(text)
    assert "Python" in skills
    assert "Machine Learning" in skills
    assert "FastAPI" in skills
    assert "Docker" in skills

def test_avoid_false_substring_matching():
    text = "I have a cat and love racing."
    skills = extract_skills(text)
    assert "C" not in skills
    assert "React" not in skills

def test_matched_and_missing_skills():
    req = {"Python", "SQL", "Docker"}
    cand = {"python", "sql", "FastAPI"}
    matched = get_matched_skills(req, cand)
    missing = get_missing_skills(req, cand)
    
    assert "Python" in matched or "python" in matched
    assert "SQL" in matched or "sql" in matched
    assert "Docker" in missing
