import pytest
from src.similarity import jaccard_similarity, skill_coverage

def test_jaccard_similarity():
    # A={python,sql}, B={python,sql} -> 1.0
    assert jaccard_similarity({"python", "sql"}, {"python", "sql"}) == 1.0
    
    # A={python}, B={java} -> 0.0
    assert jaccard_similarity({"python"}, {"java"}) == 0.0
    
    # A={python,sql}, B={python,java} -> 1/3
    assert abs(jaccard_similarity({"python", "sql"}, {"python", "java"}) - (1.0 / 3.0)) < 1e-4
    
    # Empty sets
    assert jaccard_similarity(set(), set()) == 0.0

def test_skill_coverage():
    req = {"python", "sql", "fastapi", "docker", "aws"}
    cand = {"python", "sql", "fastapi", "docker"}
    # 4 / 5 = 0.80
    assert abs(skill_coverage(req, cand) - 0.80) < 1e-4
    assert skill_coverage(set(), cand) == 1.0
