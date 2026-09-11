# HireMinds AI — Jupyter Notebook Explained

This document explains exactly how `resume_job_match_final.ipynb` works, line by line, and defines every term shown in the results.

---

## Table of Contents
1. [What the Notebook Does](#1-what-the-notebook-does)
2. [Code Walkthrough](#2-code-walkthrough)
   - 2.1 [Imports & Setup](#21-imports--setup)
   - 2.2 [File Upload Helpers](#22-file-upload-helpers)
   - 2.3 [Upload Widgets](#23-upload-widgets)
   - 2.4 [The Run Match Function](#24-the-run-match-function)
3. [Understanding the Results](#3-understanding-the-results)
   - 3.1 [Jaccard Similarity](#31-jaccard-similarity)
   - 3.2 [Skill Coverage](#32-skill-coverage)
   - 3.3 [Experience Match](#33-experience-match)
   - 3.4 [Education Match](#34-education-match)
   - 3.5 [Matched Skills](#35-matched-skills)
   - 3.6 [Missing Skills](#36-missing-skills)
   - 3.7 [Candidate Skills & Required Skills](#37-candidate-skills--required-skills)
   - 3.8 [Random Forest Prediction](#38-random-forest-prediction)
   - 3.9 [Confidence / Probability](#39-confidence--probability)
4. [Sample Output Explained](#4-sample-output-explained)

---

## 1. What the Notebook Does

The notebook provides an **interactive UI** to:
1. Upload your **resume** (PDF or TXT)
2. Upload the **job description** (TXT)
3. Click **Run Match** to instantly see:
   - Which skills you have vs. what the job needs
   - How well your experience and education match
   - A `SUITABLE` or `NOT SUITABLE` verdict from a trained AI model

Everything runs **locally on your machine** — no data is sent anywhere.

---

## 2. Code Walkthrough

The notebook has a **single code cell** (to avoid any ordering issues). Here is what each block inside it does:

---

### 2.1 Imports & Setup

```python
import os, io, sys
import ipywidgets as widgets
from IPython.display import display, Markdown
```

| Import | Purpose |
|--------|---------|
| `os` | File path operations (checking extension, joining paths) |
| `io` | Creates in-memory file-like objects from uploaded bytes |
| `sys` | Adds the project root folder to Python's search path |
| `ipywidgets` | Provides interactive UI elements (buttons, file uploaders) |
| `IPython.display` | Renders markdown tables and formatted text in the notebook |

```python
project_root = os.path.abspath('.')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
```

This ensures Python can find the project's `src/` modules (like `pdf_parser`, `features`, `predict`) even when the notebook is opened from a different working directory.

```python
from src.pdf_parser import extract_text_from_pdf
from src.features import extract_and_build_features
from src.predict import predict_candidate
```

These three functions are the entire ML pipeline:
- `extract_text_from_pdf` — reads text from a PDF file
- `extract_and_build_features` — converts raw text into numbers the model understands
- `predict_candidate` — feeds those numbers to the trained Random Forest model

---

### 2.2 File Upload Helpers

#### `_get_file_info(uploader)`

```python
def _get_file_info(uploader):
    val = uploader.value
    if isinstance(val, dict):
        fname = next(iter(val))       # filename is the dict key
        content = val[fname]['content']
    else:
        item = val[0]                 # older ipywidgets: tuple of Bunch objects
        content = getattr(item, 'content', None)
        fname = ...                   # extracted from metadata
    if isinstance(content, memoryview):
        content = bytes(content)      # convert buffer to plain bytes
    return content, fname
```

**Why this is needed:** Different versions of `ipywidgets` store the uploaded file differently:

| ipywidgets version | `.value` format | How we read it |
|--------------------|----------------|----------------|
| **≥ 8** (newer) | `dict` → `{"filename.pdf": {"content": ..., ...}}` | Key = filename, value = content |
| **< 8** (older) | `tuple` of `Bunch` objects | `val[0].content`, `val[0].metadata.name` |

After extracting the content, we check if it is a `memoryview` object (a raw memory buffer that Python sometimes uses internally) and convert it to plain `bytes` so we can read it like a normal file.

---

#### `load_resume_bytes(content_bytes, filename)`

```python
def load_resume_bytes(content_bytes, filename):
    if isinstance(content_bytes, memoryview):
        content_bytes = bytes(content_bytes)
    ext = os.path.splitext(filename)[1].lower()   # e.g. ".pdf" or ".txt"
    if ext == '.pdf':
        # Save to disk temporarily so pdfplumber can read it
        tmp = os.path.join('data', 'resumes', '_uploaded_resume.pdf')
        with open(tmp, 'wb') as f:
            f.write(content_bytes)
        return extract_text_from_pdf(tmp)
    else:
        return content_bytes.decode('utf-8', errors='ignore')
```

**Steps:**
1. Checks the file extension from the filename.
2. If `.pdf` — saves the bytes to a temporary file on disk (because `pdfplumber` needs a real file path, not an in-memory buffer), then extracts all text from it.
3. If `.txt` — directly decodes the bytes to a Python string.

---

#### `load_job_bytes(content_bytes, filename)`

```python
def load_job_bytes(content_bytes, filename):
    if isinstance(content_bytes, memoryview):
        content_bytes = bytes(content_bytes)
    return content_bytes.decode('utf-8', errors='ignore')
```

Job descriptions are always plain text, so this simply decodes the raw bytes. The `errors='ignore'` flag skips any characters that can't be decoded (e.g. special symbols), preventing crashes.

---

### 2.3 Upload Widgets

```python
resume_uploader = widgets.FileUpload(
    accept='.pdf,.txt', multiple=False, description='Upload Resume'
)
job_uploader = widgets.FileUpload(
    accept='.txt', multiple=False, description='Upload Job Description'
)
```

These create the clickable upload buttons visible in the notebook:
- `accept` restricts which file types the browser's file-picker shows.
- `multiple=False` ensures only one file can be selected at a time.
- When a file is selected, its binary content is stored in `uploader.value`.

```python
output_area = widgets.Output()
```

A hidden container widget. All text and tables from `run_match()` are printed inside this area. Using `Output()` keeps the results tidy below the button rather than scattered across the notebook.

---

### 2.4 The Run Match Function

```python
def run_match(btn):
    output_area.clear_output()          # clear previous results
    with output_area:
        if not resume_uploader.value or not job_uploader.value:
            print('Please upload both files.')
            return
```

Each time the button is clicked, old results are cleared first. If either widget is empty (no file chosen), it prints a reminder and exits early.

```python
        resume_bytes, resume_name = _get_file_info(resume_uploader)
        job_bytes, job_name = _get_file_info(job_uploader)
        resume_text = load_resume_bytes(resume_bytes, resume_name)
        job_text = load_job_bytes(job_bytes, job_name)
```

Extracts the raw file bytes and filename for each upload, then converts them to plain text strings.

```python
        features, details = extract_and_build_features(resume_text, job_text)
```

This is the core of the pipeline. It:
1. Extracts skills from both texts
2. Computes Jaccard Similarity and Skill Coverage
3. Extracts years of experience from both texts and compares them
4. Detects education levels and compares them
5. Returns two objects:
   - `features` — a dictionary of 4 numbers the model needs
   - `details` — the raw skill sets for display

```python
        prediction = predict_candidate(features)
```

Passes the 4 numbers into the saved Random Forest model and gets back a label and a probability score.

```python
        md = (
            f'## {emoji} Prediction: **{label}** ({prob:.1f}% confidence)\n\n'
            f'| Metric | Value |\n'
            ...
        )
        display(Markdown(md))
```

Builds a markdown string with the results and renders it as a formatted table and heading inside the `output_area`.

---

## 3. Understanding the Results

After clicking **Run Match**, you will see a results block. Here is what every term means:

---

### 3.1 Jaccard Similarity

**What it measures:** How much overlap there is between the candidate's skills and the job's required skills, relative to the combined total.

**Formula:**

```
Jaccard Similarity = |Matched Skills| / |All unique skills combined|
```

**Example:**
- Candidate skills: `{Python, SQL, React, Git}`
- Required skills: `{Python, SQL, Docker}`
- Matched: `{Python, SQL}` → 2 skills
- All unique: `{Python, SQL, React, Git, Docker}` → 5 skills
- **Jaccard = 2 / 5 = 40%**

**Interpretation:**
| Score | Meaning |
|-------|---------|
| 70–100% | Strong overlap — very good fit |
| 40–70% | Moderate overlap — decent fit |
| 0–40% | Low overlap — significant skill gaps |

> **Note:** Jaccard penalises extra skills the candidate has that the job doesn't ask for. It is a strict measure of mutual relevance.

---

### 3.2 Skill Coverage

**What it measures:** What percentage of the job's required skills the candidate actually has.

**Formula:**

```
Skill Coverage = |Matched Skills| / |Required Skills|
```

**Example:**
- Matched: `{Python, SQL}` → 2 skills
- Required: `{Python, SQL, Docker}` → 3 skills
- **Coverage = 2 / 3 = 66.7%**

**Interpretation:**
| Score | Meaning |
|-------|---------|
| 80–100% | Candidate covers most required skills |
| 50–80% | Candidate is partially qualified |
| 0–50% | Candidate is missing many required skills |

> **Skill Coverage is usually more important than Jaccard Similarity** because the job cares about whether *its* skills are met, not about how many bonus skills the candidate has.

---

### 3.3 Experience Match

**What it measures:** Whether the candidate has enough years of experience compared to what the job requires.

**How it works:**
1. A regular expression scans both texts for patterns like `"5 years"`, `"3+ years"`, `"2 years of experience"`.
2. The numbers are compared.

**Result:**
- `100%` — Candidate's years ≥ required years (or years not mentioned in either text)
- `0%` — Candidate's years < required years

**Example:**
- Resume says: `"4 years of experience in software development"`
- Job says: `"Minimum 3 years required"`
- 4 ≥ 3 → **Experience Match = 100%**

---

### 3.4 Education Match

**What it measures:** Whether the candidate's education level meets the minimum required by the job.

**How it works:**
Education levels are mapped to a numeric tier:

| Level | Tier |
|-------|------|
| High School / Diploma | 1 |
| Bachelor's / B.Tech / B.Sc | 2 |
| Master's / M.Tech / MBA | 3 |
| PhD / Doctorate | 4 |

The candidate's tier is compared with the job's required tier.

**Result:**
- `100%` — Candidate's tier ≥ required tier (or education not mentioned)
- `0%` — Candidate's tier < required tier

**Example:**
- Resume says: `"Master of Science in Computer Science"`
- Job says: `"Bachelor's degree required"`
- Tier 3 ≥ Tier 2 → **Education Match = 100%**

---

### 3.5 Matched Skills

The skills that appear in **both** the resume and the job description.

These are the skills you already have that the employer is looking for. A higher number here is always better.

**Example:**
- Resume: Python, SQL, React, Git
- Job: Python, SQL, Docker
- **Matched:** Python, SQL

---

### 3.6 Missing Skills

The skills the job requires that are **not found** in the resume.

These are the gaps — skills you would need to learn or highlight differently in your resume to improve your match score.

**Example:**
- Required: Python, SQL, Docker
- Candidate has: Python, SQL
- **Missing:** Docker

---

### 3.7 Candidate Skills & Required Skills

**Candidate Skills** — all technical keywords detected in the resume.  
**Required Skills** — all technical keywords detected in the job description.

The skill detection uses a curated list of ~150 common technical skills (e.g. Python, JavaScript, React, Docker, AWS, SQL, TensorFlow, etc.) and checks which ones appear in the text.

---

### 3.8 Random Forest Prediction

The final verdict from the trained AI model.

**What is a Random Forest?**  
A Random Forest is an ensemble of many decision trees. Each tree votes on whether the candidate is suitable, and the majority vote wins.

```
Feature vector  →  Tree 1: SUITABLE
[Jaccard=0.57]  →  Tree 2: SUITABLE    →  Majority: SUITABLE ✅
[Coverage=0.80] →  Tree 3: NOT SUITABLE
[Exp=1.0]       →  Tree 4: SUITABLE
[Edu=1.0]       →  Tree 5: SUITABLE
```

The model was trained on 1,500 labelled resume-job pairs. It learned which combinations of the 4 features tend to correspond to suitable vs. non-suitable candidates.

| Output | Meaning |
|--------|---------|
| ✅ `SUITABLE` | The model predicts this candidate is a good fit for the role |
| ❌ `NOT SUITABLE` | The model predicts there is a significant mismatch |

---

### 3.9 Confidence / Probability

The percentage shown next to the prediction (e.g. `82.0% confidence`) is the **proportion of trees** in the forest that voted for the winning label.

**Example:**
- 100 trees total
- 82 voted `SUITABLE`, 18 voted `NOT SUITABLE`
- **Confidence = 82%**

| Confidence | Meaning |
|------------|---------|
| 90–100% | Very strong signal — model is highly certain |
| 70–90% | Good confidence |
| 50–70% | Borderline — close call |
| < 50% | Not possible (the winning label always has > 50%) |

> A prediction of `SUITABLE` with only 51% confidence means the model is barely leaning one way — treat it as a borderline case.

---

## 4. Sample Output Explained

```
✅ Prediction: SUITABLE (82.0% confidence)

| Metric             | Value  |
|--------------------|--------|
| Jaccard Similarity | 57.1%  |
| Skill Coverage     | 75.0%  |
| Experience Match   | 100.0% |
| Education Match    | 100.0% |

✔ Matched Skills: Machine Learning, Python, SQL
✘ Missing Skills: Docker
Candidate Skills: Git, Machine Learning, Python, React, SQL
Required Skills: Docker, Machine Learning, Python, SQL
```

**Reading this result:**

1. **SUITABLE at 82%** — the model is fairly confident this candidate qualifies.
2. **Jaccard 57.1%** — moderate skill overlap (candidate has some extra skills like React, Git that the job doesn't need).
3. **Skill Coverage 75%** — the candidate covers 3 out of 4 required skills (missing Docker).
4. **Experience 100%** — years of experience meet the requirement.
5. **Education 100%** — education level meets the requirement.
6. **Missing: Docker** — the one gap to address; adding Docker to skills or gaining experience with it would push the score higher.

**Actionable insight:** The candidate is a good fit overall. Upskilling in Docker (or highlighting any existing container experience) would make them an even stronger match.

---

*HireMinds AI — Jupyter Notebook Documentation*
