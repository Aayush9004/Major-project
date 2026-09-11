# HireMinds AI — Classical Resume–Job Matching & Candidate Classification

## 1. Objective

Build only the algorithmic core of HireMinds AI. Do not implement a frontend, API, SQL database, LLM, embeddings, or vector database at this stage.

Inputs:
1. Candidate resume, preferably PDF.
2. Job description as TXT/plain text.

Outputs:
- Extracted skills
- Matched and missing skills
- Jaccard similarity
- Skill coverage
- Experience match
- Education match
- Structured feature vector
- Random Forest prediction
- Model probability
- Interpretable console report

Core pipeline:

```text
Resume + Job Description
          |
          v
   Text Preprocessing
          |
     +----+-------------+
     |                  |
     v                  v
Skill Extraction   Experience/Education
     |                  |
     v                  |
Jaccard Similarity      |
     |                  |
     +--------+---------+
              |
              v
      Feature Engineering
              |
              v
        Random Forest
              |
              v
     Suitable / Not Suitable
```

## 2. Technology

Use Python 3.10+.

Packages:
- pandas
- numpy
- scikit-learn
- nltk
- pdfplumber
- joblib
- matplotlib
- seaborn

Install:

```bash
pip install pandas numpy scikit-learn nltk pdfplumber joblib matplotlib seaborn
```

Download NLTK resources:

```python
import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
```

Keep dependencies minimal.

## 3. Project Structure

```text
HireMinds-ML/
├── data/
│   ├── resumes/
│   ├── jobs/
│   └── dataset.csv
├── models/
│   └── random_forest.pkl
├── src/
│   ├── __init__.py
│   ├── pdf_parser.py
│   ├── preprocessing.py
│   ├── skill_extractor.py
│   ├── experience_extractor.py
│   ├── education_extractor.py
│   ├── similarity.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_skill_extractor.py
│   └── test_similarity.py
├── main.py
├── requirements.txt
└── README.md
```

## 4. pdf_parser.py

Use `pdfplumber`.

Required function:

```python
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract readable text from all pages of a PDF resume."""
```

Requirements:
- Read every page.
- Handle pages returning `None`.
- Return one combined string.
- Raise useful errors for unreadable files.
- Do not perform NLP here.

## 5. preprocessing.py

Implement:

```python
def clean_text(text: str) -> str:
    """Normalize resume/job-description text."""
```

Steps:
1. Lowercase.
2. Remove URLs.
3. Remove email addresses.
4. Remove unnecessary punctuation.
5. Normalize whitespace.
6. Tokenize.
7. Remove English stopwords.
8. Lemmatize.
9. Return cleaned text.

Also implement:

```python
def tokenize_text(text: str) -> list[str]:
    ...
```

Be careful not to destroy technical terms such as `c++`, `c#`, `node.js`, `.net`, `sql`, and `machine learning`.

## 6. skill_extractor.py

Use a configurable predefined skill dictionary for the first version.

Include at least:

```python
SKILLS = [
    "python", "java", "c", "c++", "c#", "javascript", "typescript",
    "html", "css", "react", "node.js", "express", "django", "flask",
    "fastapi", "sql", "mysql", "postgresql", "mongodb",
    "machine learning", "deep learning", "artificial intelligence",
    "natural language processing", "nlp", "tensorflow", "pytorch",
    "scikit-learn", "pandas", "numpy", "docker", "kubernetes",
    "git", "github", "aws", "azure", "gcp"
]
```

Implement:

```python
def extract_skills(text: str, skills: list[str]) -> set[str]:
    ...
```

Use case-insensitive matching, support multi-word skills, avoid duplicates, and reduce false substring matches.

Also implement:

```python
def get_missing_skills(required_skills, candidate_skills) -> set[str]:
    ...

def get_matched_skills(required_skills, candidate_skills) -> set[str]:
    ...
```

## 7. similarity.py

### Jaccard Similarity

Implement:

```python
def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Return Jaccard similarity in [0, 1]."""
```

Formula:

`J(A,B) = |A intersection B| / |A union B|`

If both sets are empty, return `0.0`.

### Skill Coverage

Implement:

```python
def skill_coverage(required_skills, candidate_skills) -> float:
    """Percentage of required skills found in candidate skills."""
```

Formula:

`matched required skills / total required skills`

Example:
- Required = 5
- Matched = 4
- Coverage = 0.80

Jaccard and skill coverage must remain separate features.

## 8. experience_extractor.py

Use regular expressions to recognize:
- `2 years`
- `2+ years`
- `2 yrs`
- `3 years of experience`
- `experience: 4 years`
- decimal values such as `1.5 years`

Implement:

```python
def extract_experience(text: str) -> float:
    """Return estimated years of experience, or 0 if not detected."""
```

Use the largest clearly stated experience value for the prototype.

Implement:

```python
def experience_match(candidate_years: float, required_years: float) -> float:
    """Return an experience-match score in [0, 1]."""
```

Suggested logic:

```text
If required_years <= 0: return 1.0
If candidate_years >= required_years: return 1.0
Otherwise: candidate_years / required_years
```

## 9. education_extractor.py

Recognize common education terms:

```text
b.tech, b.e, be, bachelor, b.sc, bca,
m.tech, me, master, m.sc, mca, mba, phd
```

Implement:

```python
def extract_education(text: str) -> set[str]:
    ...
```

and:

```python
def education_match(candidate_education, required_education) -> float:
    ...
```

Use simple, explainable rules.

## 10. features.py

Combine extracted information into a structured feature vector.

Required features:

```text
jaccard_similarity
skill_coverage
experience_match
education_match
matched_skill_count
missing_skill_count
candidate_skill_count
required_skill_count
```

Define:

```python
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
```

Implement:

```python
def build_features(
    candidate_skills,
    required_skills,
    candidate_experience,
    required_experience,
    candidate_education,
    required_education
) -> dict:
    ...
```

Example output:

```python
{
    "jaccard_similarity": 0.667,
    "skill_coverage": 0.80,
    "experience_match": 1.0,
    "education_match": 1.0,
    "matched_skill_count": 4,
    "missing_skill_count": 1,
    "candidate_skill_count": 5,
    "required_skill_count": 5
}
```

Feature order must always match `FEATURE_COLUMNS`.


## 11. Dataset

Use the public **Resume-Job Fit Merged v1** dataset from Hugging Face:

`med2425/resume-job-fit-merged-v1`

It contains approximately 93,733 resume/job-description pairs and includes:
- `resume`
- `jd`
- `label`

The labels are:
- `Good Fit`
- `Potential Fit`
- `No Fit`

Dataset page:
https://huggingface.co/datasets/med2425/resume-job-fit-merged-v1

### Download

Recommended method:

```bash
pip install datasets
```

Then run:

```python
from datasets import load_dataset

dataset = load_dataset("med2425/resume-job-fit-merged-v1")

dataset["train"].to_csv("data/train.csv", index=False)
dataset["test"].to_csv("data/test.csv", index=False)
```

Expected files:

```text
data/
├── train.csv
└── test.csv
```

### Label conversion

For the first binary Random Forest implementation, convert:

```text
Good Fit      -> 1
Potential Fit -> 1
No Fit        -> 0
```

Keep the original label in the raw dataset. Create a separate processed dataset for the binary experiment.

### Dataset preprocessing

For every resume-job pair:

```text
resume + jd
    ↓
skill extraction
    ↓
experience extraction
    ↓
education extraction
    ↓
Jaccard similarity
    ↓
skill coverage
    ↓
experience match
    ↓
education match
    ↓
feature row
```

The final training data should contain the calculated features and binary target:

```csv
jaccard_similarity,skill_coverage,experience_match,education_match,matched_skill_count,missing_skill_count,candidate_skill_count,required_skill_count,label
```

Do not calculate the features manually or hardcode them.

### Important experimental rule

Do not use the test set during feature/model training.

Use the provided training split for model development and the provided test split for final evaluation. If a validation set is needed, split the training data further.

Do not claim that the dataset's labels are equivalent to a real company's hiring decision. Describe them as the dataset's fit labels.


## 12. train.py

Use:

```python
from sklearn.ensemble import RandomForestClassifier
```

Initial model:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)
```

Training procedure:
1. Load dataset.
2. Validate columns and labels.
3. Separate X and y.
4. Train/test split.
5. Use stratification when appropriate.
6. Train model.
7. Calculate accuracy, precision, recall, F1-score, and confusion matrix.
8. Save model with joblib.

```python
joblib.dump(model, "models/random_forest.pkl")
```

Do not report performance unless actually measured.

Also print feature importance for all `FEATURE_COLUMNS`.

## 13. predict.py

Load:

```python
model = joblib.load("models/random_forest.pkl")
```

Implement:

```python
def predict_candidate(features: dict) -> dict:
    ...
```

Return:

```python
{
    "prediction": "Suitable",
    "probability": 0.87
}
```

Use `predict_proba()` where available. Clearly describe this as the model's estimated class probability, not a guarantee.

## 14. main.py

Create a command-line interface.

Example:

```bash
python main.py --resume data/resumes/candidate1.pdf --job data/jobs/python_developer.txt
```

Expected output:

```text
========================================
       HireMinds AI - Candidate Match
========================================

Candidate Skills:
Python
SQL
FastAPI
Machine Learning
React

Required Skills:
Python
SQL
FastAPI
Machine Learning
Docker

Matched Skills:
Python
SQL
FastAPI
Machine Learning

Missing Skills:
Docker

----------------------------------------
Jaccard Similarity : 66.67%
Skill Coverage    : 80.00%
Experience Match  : 100.00%
Education Match   : 100.00%

----------------------------------------
Random Forest Prediction: SUITABLE
Model Probability       : 87.00%
========================================
```

The displayed numbers must be calculated dynamically.

## 15. Testing

Test preprocessing:
- Empty text
- Punctuation
- Normal text
- Technical terms

Test skill extraction:
- Case differences
- Multi-word skills
- Duplicate skills
- Missing skills

Test Jaccard:

```text
A={python,sql}, B={python,sql} -> 1.0
A={python}, B={java} -> 0.0
A={python,sql}, B={python,java} -> 1/3
```

Test experience:
- `2 years`
- `3+ years`
- `1.5 years`
- no experience

Test education:
- B.Tech
- B.E.
- M.Tech
- missing education

## 16. Example Input

Create `data/jobs/python_developer.txt`:

```text
Python Developer

We are looking for a Python Developer with at least 2 years
of experience.

Required skills:
Python, SQL, FastAPI, Machine Learning, Docker.

Education:
B.Tech or equivalent degree.
```

Example resume text:

```text
John Doe

Software Developer

Experience:
3 years of experience in software development.

Education:
B.Tech in Computer Engineering.

Skills:
Python, SQL, FastAPI, Machine Learning, React, Git.
```

Expected behavior:
- Matched skills: Python, SQL, FastAPI, Machine Learning
- Missing skill: Docker
- Skill coverage: 4/5 = 0.80
- Jaccard: calculated dynamically from extracted sets
- Experience match: 1.0
- Education match: 1.0

Do not hardcode these values.

## 17. Error Handling

Gracefully handle:
- Missing/corrupt PDF
- Empty resume
- Empty job description
- No recognized skills
- No required skills
- Missing model
- Missing dataset
- Invalid labels
- Dataset containing only one class

Use meaningful exceptions/messages.

## 18. Explainability

Every prediction should display:
- Matched skills
- Missing skills
- Jaccard similarity
- Skill coverage
- Experience match
- Education match
- Random Forest prediction
- Model probability

Training should also expose Random Forest feature importances.

## 19. Evaluation

Report:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Prefer precision, recall, and F1-score over accuracy alone when classes are imbalanced.

Do not claim the system is unbiased or production-ready.

## 20. Baseline Comparison

Implement a separate baseline:

```text
Resume + Job Description
          |
        TF-IDF
          |
  Cosine Similarity
          |
    Similarity Score
```

This is for experimental comparison, not the main proposed classifier.

Compare:

### Baseline
TF-IDF + Cosine Similarity

### Proposed classical approach
Skill Extraction + Jaccard + Skill Coverage + Experience Match + Education Match + Random Forest

This provides an experimental comparison for the project report.

## 21. Do Not Implement Yet

Do not add:
- Frontend
- FastAPI
- MySQL/PostgreSQL
- LLM
- BERT
- Sentence Transformers
- Embeddings
- Vector database
- FAISS
- Pinecone
- Chroma
- Microservices

These can be future extensions.

## 22. Development Order

Implement and test in this exact order:

1. `pdf_parser.py`
2. `preprocessing.py`
3. `skill_extractor.py`
4. `experience_extractor.py`
5. `education_extractor.py`
6. `similarity.py`
7. `features.py`
8. Create/validate labeled dataset
9. `train.py`
10. `predict.py`
11. `main.py`
12. Unit tests
13. TF-IDF + cosine baseline

Do not move to the next component while the previous component is failing its tests.

## 24. Research Positioning

Do not claim that Jaccard + Random Forest is globally novel.

Use a defensible statement such as:

> The proposed prototype combines explicit skill-set similarity, skill coverage, experience matching, education matching, and Random Forest classification into an interpretable candidate-job matching pipeline. This exact combination was not identified in the selected reference papers used for this project.

The research-paper comparison must remain evidence-based.

The dataset source and label-conversion procedure must also be documented clearly in the final report.

## 25. Final Goal

The final command:

```bash
python main.py --resume <resume.pdf> --job <job.txt>
```

must produce:
1. Candidate skills
2. Required skills
3. Matched skills
4. Missing skills
5. Jaccard similarity
6. Skill coverage
7. Experience match
8. Education match
9. Feature vector
10. Random Forest prediction
11. Prediction probability
12. Clear console output

The first version must prioritize correctness, simplicity, modularity, reproducibility, and explainability.

Future versions may add semantic embeddings/vector search and LLM-based explanations, but those are outside the current scope.
