import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data dependencies are present
for res in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    try:
        nltk.download(res, quiet=True)
    except Exception:
        pass

try:
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))

LEMMATIZER = WordNetLemmatizer()

def tokenize_text(text: str) -> list[str]:
    """
    Tokenize text into lowercased tokens while preserving technical terms like
    c++, c#, node.js, .net, sql, etc.
    """
    if not text or not isinstance(text, str):
        return []

    text_lower = text.lower()
    
    # Remove URLs
    text_clean = re.sub(r"https?://\S+|www\.\S+", " ", text_lower)
    # Remove Email addresses
    text_clean = re.sub(r"\S+@\S+", " ", text_clean)
    
    # Token pattern that handles c++, c#, .net, word.word (e.g. node.js, react.js), and normal words/numbers
    # Pattern explanation:
    # \b[a-z0-9]+(?:\.[a-z0-9]+)+\b -> domain/framework terms like node.js, react.js, .net (with preceding dot optional)
    # \.net\b                      -> .net
    # [a-z0-9]+[+#]{1,2}           -> c++, c#
    # \b[a-z0-9]+\b                -> standard word tokens
    pattern = r"\b[a-z0-9]+(?:\.[a-z0-9]+)+\b|\.net\b|[a-z0-9]+[+#]{1,2}|\b[a-z0-9]+\b"
    tokens = re.findall(pattern, text_clean)
    
    return tokens

def clean_text(text: str) -> str:
    """
    Normalize resume/job-description text:
    1. Lowercase
    2. Remove URLs & Emails
    3. Remove unnecessary punctuation while preserving tech terms
    4. Normalize whitespace
    5. Tokenize
    6. Remove English stopwords
    7. Lemmatize
    8. Return cleaned text
    """
    if not text or not isinstance(text, str):
        return ""

    tokens = tokenize_text(text)
    
    # Remove stopwords and apply lemmatization for standard words
    cleaned_tokens = []
    for token in tokens:
        if token in STOP_WORDS:
            continue
        # Avoid lemmatizing technical terms ending in symbols or numbers or dots
        if not re.search(r"[+#\.]", token):
            lemmatized = LEMMATIZER.lemmatize(token)
            cleaned_tokens.append(lemmatized)
        else:
            cleaned_tokens.append(token)
            
    return " ".join(cleaned_tokens)
