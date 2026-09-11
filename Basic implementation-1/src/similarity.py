def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """
    Calculate Jaccard similarity coefficient between two sets.
    
    Formula:
        J(A, B) = |A intersection B| / |A union B|
        
    Returns:
        float: Jaccard similarity score in [0.0, 1.0].
    """
    if not set_a and not set_b:
        return 0.0

    lower_a = {s.lower() for s in set_a}
    lower_b = {s.lower() for s in set_b}

    intersection = lower_a.intersection(lower_b)
    union = lower_a.union(lower_b)

    if not union:
        return 0.0

    return len(intersection) / len(union)

def skill_coverage(required_skills: set[str], candidate_skills: set[str]) -> float:
    """
    Calculate Skill Coverage metric:
        percentage of required skills present in candidate skills.
        
    Formula:
        matched required skills / total required skills
        
    Returns:
        float: Skill coverage score in [0.0, 1.0].
    """
    if not required_skills:
        return 1.0

    req_lower = {s.lower() for s in required_skills}
    cand_lower = {s.lower() for s in candidate_skills}

    matched = req_lower.intersection(cand_lower)

    return len(matched) / len(req_lower)
