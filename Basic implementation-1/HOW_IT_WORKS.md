# HireMinds AI — Resume & Job Description Matcher
## How It Works

---

## Table of Contents
1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [Pipeline Walkthrough](#3-pipeline-walkthrough)
   - 3.1 [PDF / Text Parsing](#31-pdf--text-parsing)
   - 3.2 [Text Preprocessing](#32-text-preprocessing)
   - 3.3 [Skill Extraction](#33-skill-extraction)
   - 3.4 [Experience & Education Matching](#34-experience--education-matching)
   - 3.5 [Feature Engineering](#35-feature-engineering)
   - 3.6 [Model Inference](#36-model-inference)
4. [Interactive Notebook](#4-interactive-notebook)
5. [Command-Line Interface](#5-command-line-interface)
6. [Project Structure](#6-project-structure)
7. [Model Training](#7-model-training)
8. [Dependencies](#8-dependencies)

---

## 1. Overview

**HireMinds AI** is a classical machine-learning pipeline that takes a candidate's **resume** and a **job description** and predicts whether the candidate is `SUITABLE` or `NOT SUITABLE` for the role.

It uses a **Random Forest classifier** trained on the [med2425/resume-job-fit-merged-v1](https://huggingface.co/datasets/med2425/resume-job-fit-merged-v1) dataset (1,500 training / 400 test samples).

The prediction is based on four hand-engineered signals:

| Feature | Description |
|---------|-------------|
| **Jaccard Similarity** | Overlap between candidate and required skill sets |
| **Skill Coverage** | % of required skills the candidate possesses |
| **Experience Match** | Whether years of experience meet the job's requirement |
| **Education Match** | Whether the candidate's education level meets the role |

---

## 2. System Architecture

```
                       ┌─────────────────────────────────┐
                       │        User Input                │
                       │  Resume (PDF/TXT) + Job (TXT)    │
                       └────────────┬────────────────────┘
                                    │
                       ┌────────────▼────────────────────┐
                       │       PDF Parser                 │
                       │   src/pdf_parser.py              │
                       │   Extracts raw text from PDF     │
                       └────────────┬────────────────────┘
                                    │
                       ┌────────────▼────────────────────┐
                       │      Text Preprocessor           │
                       │   src/preprocessing.py           │
                       │   Lowercasing, stop-word removal │
                       └────────────┬────────────────────┘
                                    │
              ┌─────────────────────┼──────────────────────┐
              │                     │                       │
   ┌──────────▼────────┐  ┌─────────▼─────────┐  ┌────────▼────────┐
   │  Skill Extractor  │  │ Experience Extractor│  │ Education Check │
   │ src/skill_extractor│  │src/experience_extractor│  │src/education_extractor│
   └──────────┬────────┘  └─────────┬─────────┘  └────────┬────────┘
              │                     │                       │
              └─────────────────────┼───────────────────────┘
                                    │
                       ┌────────────▼────────────────────┐
                       │     Feature Engineering          │
                       │     src/features.py              │
                       │  Computes 4 numeric features     │
                       └────────────┬────────────────────┘
                                    │
                       ┌────────────▼────────────────────┐
                       │     Random Forest Classifier     │
                       │     src/predict.py               │
                       │     models/random_forest.pkl     │
                       └────────────┬────────────────────┘
                                    │
                       ┌────────────▼────────────────────┐
                       │     Result Output                │
                       │  SUITABLE / NOT SUITABLE + %    │
                       └─────────────────────────────────┘
```

---

## 3. Pipeline Walkthrough

### 3.1 PDF / Text Parsing
**File:** `src/pdf_parser.py`

If a **PDF** is supplied, the pipeline uses [`pdfplumber`](https://github.com/jsvine/pdfplumber) to extract readable text from every page. Plain `.txt` files are read directly. The output is a single combined string.

```python
extract_text_from_pdf("data/resumes/my_resume.pdf")
# → "John Doe\nSoftware Engineer\n5 years experience..."
```

---

### 3.2 Text Preprocessing
**File:** `src/preprocessing.py`

The raw text is cleaned:
- Converted to lowercase
- Punctuation removed
- Common stop-words removed (e.g., "the", "and", "or")
- Whitespace normalised

```python
clean_text("Experienced in Python, SQL & Machine Learning!")
# → "experienced python sql machine learning"
```

---

### 3.3 Skill Extraction
**File:** `src/skill_extractor.py`

A **keyword-matching** approach is used against a curated list of ~150 technical skills (e.g., `Python`, `React`, `Docker`, `TensorFlow`, `SQL`, etc.).

The extractor scans the cleaned text and returns a set of matching skill tokens for both the **resume** and the **job description**.

```python
extract_skills("python sql machine learning docker")
# → {"python", "sql", "machine learning", "docker"}
```

The intersection of the two sets gives **matched skills**; the difference gives **missing skills**.

---

### 3.4 Experience & Education Matching
**Files:** `src/experience_extractor.py`, `src/education_extractor.py`

**Experience Matching**  
Uses regular expressions to detect patterns like `"5 years"`, `"3+ years"` in both documents and compares the numbers:
- Returns `1.0` if candidate years ≥ required years (or if neither document mentions years)
- Returns `0.0` if candidate years < required years

**Education Matching**  
Detects education levels (`Bachelor`, `Master`, `PhD`, `MBA`, `Diploma`, `High School`) and maps them to a numeric tier. Returns `1.0` if the candidate meets or exceeds the required tier; `0.0` otherwise.

---

### 3.5 Feature Engineering
**File:** `src/features.py`

All extracted signals are combined into a 4-dimensional feature vector:

| Feature | Formula |
|---------|---------|
| `jaccard_similarity` | \|matched_skills\| / \|candidate_skills ∪ required_skills\| |
| `skill_coverage` | \|matched_skills\| / \|required_skills\| |
| `experience_match` | 1.0 or 0.0 (from extractor) |
| `education_match` | 1.0 or 0.0 (from extractor) |

```python
features = {
    "jaccard_similarity": 0.57,
    "skill_coverage":     0.80,
    "experience_match":   1.0,
    "education_match":    1.0
}
```

This dictionary is also returned alongside a `details` dict containing the raw skill sets for display.

---

### 3.6 Model Inference
**File:** `src/predict.py`  
**Model:** `models/random_forest.pkl`

The 4 features are passed as a 1-row NumPy array to a pre-trained **RandomForestClassifier** (scikit-learn). The model returns:
- **`prediction`** — `"SUITABLE"` or `"NOT SUITABLE"`
- **`probability`** — confidence score (0.0 – 1.0)

```python
predict_candidate(features)
# → {"prediction": "SUITABLE", "probability": 0.82}
```

> The model was trained on 1,500 samples from the Hugging Face dataset `med2425/resume-job-fit-merged-v1` with an 80/20 train/test split.

---

## 4. Interactive Notebook

**File:** `resume_job_match_final.ipynb`

The notebook provides a **one-cell interactive UI** built with `ipywidgets`.

### How to launch

```powershell
jupyter lab resume_job_match_final.ipynb
```

### Steps inside the notebook

| Step | Action |
|------|--------|
| **1** | Run the single code cell (Shift+Enter) |
| **2** | Click **Upload Resume** → select your PDF or TXT resume |
| **3** | Click **Upload Job Description** → select your job TXT file |
| **4** | Click the green **Run Match** button |
| **5** | A formatted table with skills and prediction appears below |

### Compatibility notes

The notebook handles both versions of `ipywidgets`:

| Version | `.value` type | Handled by |
|---------|--------------|------------|
| ≥ 8 | `dict` keyed by filename | `_get_file_info()` |
| < 8 | `tuple` of `Bunch` objects | `_get_file_info()` |
| Either | `memoryview` content | converted via `bytes(content)` |

---

## 5. Command-Line Interface

**File:** `main.py`

```powershell
python main.py --resume data/resumes/my_resume.pdf --job data/jobs/my_job.txt
```

Optional flag:

```powershell
python main.py --resume ... --job ... --baseline
```

The `--baseline` flag also runs a **TF-IDF cosine similarity** comparison (`src/baseline.py`) alongside the Random Forest prediction.

### Sample output

```
============================================================
          HireMinds AI — Resume Matcher Report
============================================================

Candidate Skills : Python, SQL, Machine Learning, React, Git
Required Skills  : Python, SQL, Machine Learning, Docker

Matched Skills   : Python, SQL, Machine Learning
Missing Skills   : Docker

Jaccard Similarity : 57.14%
Skill Coverage     : 75.00%
Experience Match   : 100.00%
Education Match    : 100.00%

────────────────────────────────────────────────────────────
Random Forest  →  SUITABLE  (82.00% confidence)
============================================================
```

---

## 6. Project Structure

```
Basic implementation-1/
│
├── data/
│   ├── resumes/                   # Place candidate PDFs / TXTs here
│   │   └── sample_candidate.pdf
│   └── jobs/                      # Place job descriptions (TXT) here
│       └── python_developer.txt
│
├── models/
│   └── random_forest.pkl          # Trained model (auto-generated by train.py)
│
├── src/
│   ├── __init__.py
│   ├── pdf_parser.py              # PDF → raw text
│   ├── preprocessing.py           # Clean & normalise text
│   ├── skill_extractor.py         # Keyword-based skill detection
│   ├── experience_extractor.py    # Regex-based years of experience detection
│   ├── education_extractor.py     # Education level detection
│   ├── features.py                # Builds the 4-feature vector
│   ├── similarity.py              # Jaccard & cosine utilities
│   ├── baseline.py                # TF-IDF baseline
│   ├── predict.py                 # Load model & run inference
│   └── train.py                   # Training script
│
├── tests/                         # Unit tests (pytest)
│
├── main.py                        # CLI entry point
├── prepare_dataset.py             # Downloads & prepares training data
├── create_notebook.py             # Generates resume_job_match_final.ipynb
├── resume_job_match_final.ipynb   # Interactive Jupyter UI
└── requirements.txt               # Python dependencies
```

---

## 7. Model Training

To retrain the model from scratch:

```powershell
# 1. Download and prepare the dataset
python prepare_dataset.py

# 2. Train the Random Forest classifier
python src/train.py
```

Training details:
- **Dataset:** `med2425/resume-job-fit-merged-v1` (Hugging Face)
- **Split:** 1,500 train / 400 test samples
- **Algorithm:** `RandomForestClassifier` (scikit-learn, 100 estimators)
- **Features:** 4 numeric features (see §3.5)
- **Output:** `models/random_forest.pkl`

---

## 8. Dependencies

| Package | Purpose |
|---------|---------|
| `scikit-learn` | Random Forest classifier |
| `pdfplumber` | PDF text extraction |
| `datasets` | Hugging Face dataset loading |
| `pandas` | DataFrame operations |
| `numpy` | Array operations |
| `ipywidgets` | Notebook upload widgets |
| `jupyterlab` | Interactive notebook UI |
| `pytest` | Unit tests |

Install all dependencies:

```powershell
pip install -r requirements.txt
pip install --upgrade ipywidgets jupyterlab
```

---

*HireMinds AI — Classical ML Resume Matcher | Built with scikit-learn, pdfplumber & ipywidgets*
