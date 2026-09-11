from src.skill_extractor import extract_skills, get_matched_skills, get_missing_skills
from src.experience_extractor import extract_experience, experience_match
from src.education_extractor import extract_education, education_match
from src.similarity import jaccard_similarity, skill_coverage

FEATURE_COLUMNS = [
    "jaccard_similarity",
    "skill_coverage",
    "experience_match",
    "education_match",
    "matched_skill_count",
    "missing_skill_count",
    "candidate_skill_count",
    "required_skill_count"
]

def build_features(
    candidate_skills: set[str],
    required_skills: set[str],
    candidate_experience: float,
    required_experience: float,
    candidate_education: set[str],
    required_education: set[str]
) -> dict:
    """
    Combine extracted candidate and job properties into a structured feature dictionary.
    Feature order matches FEATURE_COLUMNS exactly.
    """
    matched_skills = get_matched_skills(required_skills, candidate_skills)
    missing_skills = get_missing_skills(required_skills, candidate_skills)
    
    jaccard_score = round(jaccard_similarity(candidate_skills, required_skills), 4)
    coverage_score = round(skill_coverage(required_skills, candidate_skills), 4)
    exp_score = round(experience_match(candidate_experience, required_experience), 4)
    edu_score = round(education_match(candidate_education, required_education), 4)

    features = {
        "jaccard_similarity": jaccard_score,
        "skill_coverage": coverage_score,
        "experience_match": exp_score,
        "education_match": edu_score,
        "matched_skill_count": len(matched_skills),
        "missing_skill_count": len(missing_skills),
        "candidate_skill_count": len(candidate_skills),
        "required_skill_count": len(required_skills)
    }

    return features

def extract_and_build_features(resume_text: str, job_text: str) -> tuple[dict, dict]:
    """
    Pipeline helper: extracts skills, experience, education from raw resume & job text,
    and returns both the calculated feature dictionary and an extraction details dictionary.
    
    Returns:
        tuple[dict, dict]: (features_dict, extraction_details_dict)
    """
    cand_skills = extract_skills(resume_text)
    req_skills = extract_skills(job_text)

    cand_exp = extract_experience(resume_text)
    req_exp = extract_experience(job_text)

    cand_edu = extract_education(resume_text)
    req_edu = extract_education(job_text)

    features = build_features(
        candidate_skills=cand_skills,
        required_skills=req_skills,
        candidate_experience=cand_exp,
        required_experience=req_exp,
        candidate_education=cand_edu,
        required_education=req_edu
    )

    matched_skills = get_matched_skills(req_skills, cand_skills)
    missing_skills = get_missing_skills(req_skills, cand_skills)

    details = {
        "candidate_skills": cand_skills,
        "required_skills": req_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "candidate_experience": cand_exp,
        "required_experience": req_exp,
        "candidate_education": cand_edu,
        "required_education": req_edu
    }

    return features, details
