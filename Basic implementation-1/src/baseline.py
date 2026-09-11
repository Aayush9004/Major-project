from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing import clean_text

def compute_tfidf_baseline(resume_text: str, job_text: str) -> dict:
    """
    Compute baseline match score using TF-IDF and Cosine Similarity.
    
    Args:
        resume_text (str): Candidate resume text.
        job_text (str): Job description text.
        
    Returns:
        dict: {"baseline_similarity": float, "prediction": "SUITABLE" | "NOT SUITABLE"}
    """
    cleaned_resume = clean_text(resume_text)
    cleaned_job = clean_text(job_text)

    if not cleaned_resume or not cleaned_job:
        return {
            "baseline_similarity": 0.0,
            "prediction": "NOT SUITABLE"
        }

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([cleaned_resume, cleaned_job])

    cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    cos_sim_score = round(float(cos_sim), 4)

    # Threshold for baseline suitability classification (e.g. 0.35 similarity)
    prediction = "SUITABLE" if cos_sim_score >= 0.35 else "NOT SUITABLE"

    return {
        "baseline_similarity": cos_sim_score,
        "prediction": prediction
    }
