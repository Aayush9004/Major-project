import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from src.features import extract_and_build_features, FEATURE_COLUMNS

DATA_DIR = "data"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

LABEL_MAP = {
    "Good Fit": 1,
    "Potential Fit": 1,
    "No Fit": 0,
    "good_fit": 1,
    "potential_fit": 1,
    "no_fit": 0,
    1: 1,
    0: 0
}

def generate_synthetic_samples(num_samples: int = 200) -> pd.DataFrame:
    """Fallback generator for synthetic resume/job pairs if dataset download is unavailable."""
    print("Generating synthetic resume-job dataset for training...")
    records = []
    
    sample_skills = [
        "python", "java", "c++", "javascript", "react", "node.js",
        "sql", "fastapi", "docker", "kubernetes", "aws", "machine learning",
        "deep learning", "nlp", "pandas", "git"
    ]
    
    sample_degrees = ["B.Tech", "B.E.", "M.Tech", "M.Sc", "PhD", "BCA", "MCA"]
    
    np.random.seed(42)
    
    for i in range(num_samples):
        # Generate job
        req_sk = list(np.random.choice(sample_skills, size=np.random.randint(3, 7), replace=False))
        req_exp_val = np.random.choice([0, 1, 2, 3, 5])
        req_deg = np.random.choice(sample_degrees)
        
        jd_text = f"Job Title: Software Engineer\nExperience required: {req_exp_val} years\nEducation: {req_deg}\nRequired Skills: {', '.join(req_sk)}"
        
        # Determine if fit or no fit
        is_good = np.random.rand() > 0.4
        
        if is_good:
            # High skill match, good exp
            cand_sk = list(set(req_sk[:max(1, len(req_sk)-1)] + list(np.random.choice(sample_skills, size=2, replace=False))))
            cand_exp_val = max(req_exp_val, np.random.randint(1, 6))
            cand_deg = req_deg
            label = 1
        else:
            # Low skill match, low exp
            other_skills = [s for s in sample_skills if s not in req_sk]
            cand_sk = list(np.random.choice(other_skills, size=np.random.randint(1, 3), replace=False))
            cand_exp_val = max(0, req_exp_val - 2)
            cand_deg = "High School"
            label = 0
            
        resume_text = f"Candidate Profile\nExperience: {cand_exp_val} years in software.\nEducation: {cand_deg}\nSkills: {', '.join(cand_sk)}"
        
        records.append({
            "resume": resume_text,
            "jd": jd_text,
            "label": label
        })
        
    return pd.DataFrame(records)

def process_dataframe(df: pd.DataFrame, max_samples: int = 2000) -> pd.DataFrame:
    """Extract features from resume/jd pairs dataframe."""
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        
    feature_rows = []
    
    for idx, row in df.iterrows():
        resume_text = str(row.get("resume", row.get("resume_text", "")))
        jd_text = str(row.get("jd", row.get("job_description", "")))
        raw_label = row.get("label", 0)
        
        binary_label = LABEL_MAP.get(raw_label, 1 if str(raw_label).lower() in ["good fit", "potential fit", "1"] else 0)
        
        features, _ = extract_and_build_features(resume_text, jd_text)
        features["label"] = binary_label
        feature_rows.append(features)
        
    return pd.DataFrame(feature_rows)

def prepare_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "resumes"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "jobs"), exist_ok=True)

    print("Attempting to load dataset med2425/resume-job-fit-merged-v1 from Hugging Face...")
    try:
        dataset = load_dataset("med2425/resume-job-fit-merged-v1")
        print("Dataset loaded successfully!")
        
        if "train" in dataset:
            train_df_raw = pd.DataFrame(dataset["train"])
        else:
            train_df_raw = pd.DataFrame(dataset)
            
        if "test" in dataset:
            test_df_raw = pd.DataFrame(dataset["test"])
        else:
            # Split manually if no test split provided
            msk = np.random.rand(len(train_df_raw)) < 0.8
            test_df_raw = train_df_raw[~msk].reset_index(drop=True)
            train_df_raw = train_df_raw[msk].reset_index(drop=True)
            
    except Exception as e:
        print(f"Warning: Could not download Hugging Face dataset ({str(e)}). Falling back to synthetic generator.")
        raw_data = generate_synthetic_samples(num_samples=500)
        msk = np.random.rand(len(raw_data)) < 0.8
        train_df_raw = raw_data[msk].reset_index(drop=True)
        test_df_raw = raw_data[~msk].reset_index(drop=True)

    print("Processing feature extraction for training dataset...")
    train_features_df = process_dataframe(train_df_raw, max_samples=1500)
    print("Processing feature extraction for testing dataset...")
    test_features_df = process_dataframe(test_df_raw, max_samples=400)

    train_features_df.to_csv(TRAIN_PATH, index=False)
    test_features_df.to_csv(TEST_PATH, index=False)

    print(f"Dataset preparation complete!\n- Train dataset: {TRAIN_PATH} ({len(train_features_df)} rows)\n- Test dataset: {TEST_PATH} ({len(test_features_df)} rows)")

if __name__ == "__main__":
    prepare_data()
