import argparse
import os
import sys

from src.pdf_parser import extract_text_from_pdf
from src.features import extract_and_build_features
from src.predict import predict_candidate
from src.baseline import compute_tfidf_baseline

def load_resume_text(resume_path: str) -> str:
    if not os.path.exists(resume_path):
        raise FileNotFoundError(f"Resume file not found: {resume_path}")

    if resume_path.lower().endswith(".pdf"):
        return extract_text_from_pdf(resume_path)
    else:
        with open(resume_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

def load_job_text(job_path: str) -> str:
    if not os.path.exists(job_path):
        raise FileNotFoundError(f"Job description file not found: {job_path}")

    with open(job_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def print_report(details: dict, features: dict, prediction_result: dict, baseline_result: dict = None):
    print("\n========================================")
    print("       HireMinds AI - Candidate Match   ")
    print("========================================")

    print("\nCandidate Skills:")
    if details["candidate_skills"]:
        for sk in sorted(details["candidate_skills"]):
            print(f"  - {sk}")
    else:
        print("  (None detected)")

    print("\nRequired Skills:")
    if details["required_skills"]:
        for sk in sorted(details["required_skills"]):
            print(f"  - {sk}")
    else:
        print("  (None detected)")

    print("\nMatched Skills:")
    if details["matched_skills"]:
        for sk in sorted(details["matched_skills"]):
            print(f"  - {sk}")
    else:
        print("  (None)")

    print("\nMissing Skills:")
    if details["missing_skills"]:
        for sk in sorted(details["missing_skills"]):
            print(f"  - {sk}")
    else:
        print("  (None)")

    print("\n----------------------------------------")
    print(f"Jaccard Similarity : {features['jaccard_similarity'] * 100:.2f}%")
    print(f"Skill Coverage     : {features['skill_coverage'] * 100:.2f}%")
    print(f"Experience Match   : {features['experience_match'] * 100:.2f}%")
    print(f"Education Match    : {features['education_match'] * 100:.2f}%")

    print("\n----------------------------------------")
    print(f"Random Forest Prediction: {prediction_result['prediction']}")
    print(f"Model Probability       : {prediction_result['probability'] * 100:.2f}%")

    if baseline_result:
        print("\n----------------------------------------")
        print("TF-IDF Baseline Comparison:")
        print(f"Cosine Similarity Score : {baseline_result['baseline_similarity'] * 100:.2f}%")
        print(f"Baseline Prediction     : {baseline_result['prediction']}")

    print("========================================\n")

def main():
    parser = argparse.ArgumentParser(description="HireMinds AI - Candidate Resume & Job Matching")
    parser.add_argument("--resume", type=str, help="Path to candidate resume PDF or TXT")
    parser.add_argument("--job", type=str, help="Path to job description TXT file")
    parser.add_argument("--train", action="store_true", help="Download dataset and train Random Forest model")
    parser.add_argument("--baseline", action="store_true", help="Include TF-IDF + Cosine similarity baseline")

    args = parser.parse_args()

    if args.train:
        print("Starting dataset preparation and model training...")
        from prepare_dataset import prepare_data
        from src.train import train_model
        prepare_data()
        train_model()
        return

    if not args.resume or not args.job:
        print("Error: Please provide both --resume and --job parameters (or use --train to train the model).")
        print("Example usage:\n  python main.py --resume data/resumes/sample_candidate.pdf --job data/jobs/python_developer.txt")
        sys.exit(1)

    try:
        resume_text = load_resume_text(args.resume)
        job_text = load_job_text(args.job)
        
        if not resume_text.strip():
            print("Error: Resume file contains no text.")
            sys.exit(1)
            
        if not job_text.strip():
            print("Error: Job description file contains no text.")
            sys.exit(1)

        features, details = extract_and_build_features(resume_text, job_text)

        # Check if model exists, if not auto-train or raise informative error
        model_path = "models/random_forest.pkl"
        if not os.path.exists(model_path):
            print("Model checkpoint not found. Auto-training model first...")
            from prepare_dataset import prepare_data
            from src.train import train_model
            prepare_data()
            train_model()

        pred = predict_candidate(features, model_path=model_path)

        baseline_res = None
        if args.baseline:
            baseline_res = compute_tfidf_baseline(resume_text, job_text)

        print_report(details, features, pred, baseline_res)

    except Exception as e:
        print(f"\n[Error] {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
