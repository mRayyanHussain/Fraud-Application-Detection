import re

# ----------------------------------
# SIMPLE STOPWORDS (DEPLOY-SAFE)
# ----------------------------------
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "in", "on", "at", "for", "to", "from", "by",
    "and", "or", "but", "if", "then", "this", "that",
    "it", "of", "with", "as", "be", "have", "has",
    "had", "i", "you", "he", "she", "they", "we",
    "my", "your", "his", "her", "their", "our"
}

# ----------------------------------
# TEXT CLEANING FUNCTION
# ----------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]
    return " ".join(words)
