import re

def extract_experience(text: str) -> float:
    """
    Extract estimated years of experience from text using regular expressions.
    Returns the maximum value found, or 0.0 if not detected.
    """
    if not text or not isinstance(text, str):
        return 0.0

    text_lower = text.lower()
    
    # Patterns for experience
    # E.g. "2 years", "2+ years", "2 yrs", "3 years of experience", "experience: 4 years", "1.5 years"
    patterns = [
        r"(\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:years?|yrs?)(?:\s*of\s*experience)?",
        r"experience\s*:\s*(\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:years?|yrs?)?",
        r"(\d+(?:\.\d+)?)\s*(?:\+)?\s*years?\s*in",
    ]

    found_years = []
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            try:
                val = float(match)
                # Filter out unrealistically huge numbers (e.g. year 2024 or postal codes)
                if val < 50.0:
                    found_years.append(val)
            except ValueError:
                pass

    if not found_years:
        return 0.0
        
    return max(found_years)

def experience_match(candidate_years: float, required_years: float) -> float:
    """
    Return an experience-match score in [0.0, 1.0].
    
    Logic:
    - If required_years <= 0: return 1.0
    - If candidate_years >= required_years: return 1.0
    - Otherwise: candidate_years / required_years
    """
    try:
        cand_y = float(candidate_years)
        req_y = float(required_years)
    except (ValueError, TypeError):
        return 0.0

    if req_y <= 0.0:
        return 1.0
    if cand_y >= req_y:
        return 1.0
        
    return min(1.0, max(0.0, cand_y / req_y))
