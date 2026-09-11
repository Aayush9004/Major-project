import re

DEGREE_MAP = {
    # Doctorate (Level 3)
    "phd": 3, "doctorate": 3, "doctor": 3,
    # Master's (Level 2)
    "m.tech": 2, "mtech": 2, "m.e": 2, "me": 2, "master": 2, "masters": 2,
    "m.sc": 2, "msc": 2, "mca": 2, "mba": 2, "ms": 2,
    # Bachelor's (Level 1)
    "b.tech": 1, "btech": 1, "b.e": 1, "be": 1, "bachelor": 1, "bachelors": 1,
    "b.sc": 1, "bsc": 1, "bca": 1, "bs": 1, "ba": 1
}

def extract_education(text: str) -> set[str]:
    """
    Extract education degrees present in text.
    Returns a set of recognized normalized degree terms.
    """
    if not text or not isinstance(text, str):
        return set()

    text_lower = text.lower()
    found_degrees = set()

    for degree in DEGREE_MAP:
        escaped = re.escape(degree)
        pattern = r"(?:\b|_)" + escaped + r"(?:\b|_)"
        if re.search(pattern, text_lower):
            found_degrees.add(degree)

    return found_degrees

def get_max_education_level(degrees: set[str]) -> int:
    """Return highest education level numerical index (0=None, 1=Bachelor, 2=Master, 3=PhD)."""
    if not degrees:
        return 0
    return max([DEGREE_MAP.get(d, 0) for d in degrees], default=0)

def education_match(candidate_education: set[str], required_education: set[str]) -> float:
    """
    Return an education-match score in [0.0, 1.0].
    
    Logic:
    - If required_education is empty: return 1.0 (no strict requirement)
    - If candidate has required degree or candidate level >= required level: return 1.0
    - If candidate level is lower than required level: return 0.5 (partial match)
    - Otherwise (no candidate education found): return 0.0
    """
    if not required_education:
        return 1.0

    cand_level = get_max_education_level(candidate_education)
    req_level = get_max_education_level(required_education)

    if req_level == 0:
        return 1.0
    if cand_level >= req_level:
        return 1.0
    if cand_level > 0:
        return 0.5

    return 0.0
