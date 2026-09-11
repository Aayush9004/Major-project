import re
from src.preprocessing import tokenize_text

SKILLS = [
    "python", "java", "c", "c++", "c#", "javascript", "typescript",
    "html", "css", "react", "node.js", "express", "django", "flask",
    "fastapi", "sql", "mysql", "postgresql", "mongodb",
    "machine learning", "deep learning", "artificial intelligence",
    "natural language processing", "nlp", "tensorflow", "pytorch",
    "scikit-learn", "pandas", "numpy", "docker", "kubernetes",
    "git", "github", "aws", "azure", "gcp"
]

# Canonical display dictionary mapping lowercase skill to proper display format
SKILL_DISPLAY_MAP = {
    "python": "Python",
    "java": "Java",
    "c": "C",
    "c++": "C++",
    "c#": "C#",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "html": "HTML",
    "css": "CSS",
    "react": "React",
    "node.js": "Node.js",
    "express": "Express",
    "django": "Django",
    "flask": "Flask",
    "fastapi": "FastAPI",
    "sql": "SQL",
    "mysql": "MySQL",
    "postgresql": "PostgreSQL",
    "mongodb": "MongoDB",
    "machine learning": "Machine Learning",
    "deep learning": "Deep Learning",
    "artificial intelligence": "Artificial Intelligence",
    "natural language processing": "Natural Language Processing",
    "nlp": "NLP",
    "tensorflow": "TensorFlow",
    "pytorch": "PyTorch",
    "scikit-learn": "Scikit-Learn",
    "pandas": "Pandas",
    "numpy": "NumPy",
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "git": "Git",
    "github": "GitHub",
    "aws": "AWS",
    "azure": "Azure",
    "gcp": "GCP"
}

def extract_skills(text: str, skills: list[str] = SKILLS) -> set[str]:
    """
    Extract skills present in text using precise matching to avoid false positives.
    
    Args:
        text (str): Input text (resume or job description).
        skills (list[str]): List of skills to extract.
        
    Returns:
        set[str]: Set of matched skill names (canonical casing if known, or original skill string).
    """
    if not text or not isinstance(text, str):
        return set()

    text_lower = text.lower()
    tokens = set(tokenize_text(text))
    extracted = set()

    for skill in skills:
        skill_lower = skill.lower()
        
        if " " in skill_lower or "-" in skill_lower or "." in skill_lower:
            # Multi-word or multi-character token skills (e.g. "machine learning", "scikit-learn", "node.js")
            # Use regex pattern with word/character boundary checking
            escaped = re.escape(skill_lower)
            pattern = r"(?:\b|_)" + escaped + r"(?:\b|_)"
            if re.search(pattern, text_lower):
                extracted.add(SKILL_DISPLAY_MAP.get(skill_lower, skill))
        else:
            # Single-word skill (e.g. "python", "c", "java", "sql")
            # Check exact presence in tokenized set
            if skill_lower in tokens:
                extracted.add(SKILL_DISPLAY_MAP.get(skill_lower, skill))

    return extracted

def get_missing_skills(required_skills: set[str], candidate_skills: set[str]) -> set[str]:
    """
    Return skills required by the job that are missing in the candidate skills.
    Matches are case-insensitive.
    """
    candidate_lower = {s.lower() for s in candidate_skills}
    missing = set()
    for req in required_skills:
        if req.lower() not in candidate_lower:
            missing.add(req)
    return missing

def get_matched_skills(required_skills: set[str], candidate_skills: set[str]) -> set[str]:
    """
    Return skills required by the job that are present in candidate skills.
    Matches are case-insensitive.
    """
    candidate_lower = {s.lower() for s in candidate_skills}
    matched = set()
    for req in required_skills:
        if req.lower() in candidate_lower:
            matched.add(req)
    return matched
