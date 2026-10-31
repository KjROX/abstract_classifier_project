"""NLTK-based abstract cleaning with offline-safe stopword fallback."""
import re
from nltk.tokenize import wordpunct_tokenize

def clean_text(text):
    text = "" if text is None else str(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text.lower())
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z\s'-]", " ", text)
    tokens = wordpunct_tokenize(text)  # NLTK tokenizer does not require downloaded corpora.
    try:
        from nltk.corpus import stopwords
        stops = set(stopwords.words("english"))
    except LookupError:
        # Offline-safe fallback; NLTK tokenization remains in use.
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        stops = set(ENGLISH_STOP_WORDS)
    stops -= {"no", "nor", "not", "against", "without"}
    return " ".join(t for t in tokens if len(t) > 1 and t not in stops)
