"""
src/nlp_utils.py
----------------
Utility functions for multilingual NLP pipeline:
- Text cleaning
- Tokenization
- N-gram modeling & perplexity
- POS tagging
- Dependency parsing visualization
"""

import re
import math
import contractions
import spacy
from nltk import ngrams
from collections import Counter
from spacy import displacy


# =====================================================
# 🧹 TEXT CLEANING
# =====================================================
def clean_text(text: str) -> str:
    """Cleans text by lowercasing, expanding contractions, removing symbols."""
    text = str(text).lower()
    text = contractions.fix(text)
    text = re.sub(r'[-/]', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r"[^a-zA-Záéíóúüñ'\s]", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# =====================================================
# ✂️ TOKENIZATION
# =====================================================
def tokenize_text(doc):
    """Tokenizes a spaCy Doc into a list of tokens."""
    return [token.text for token in doc if not token.is_punct and not token.is_space]


# =====================================================
# 🧮 N-GRAM MODELS & PERPLEXITY
# =====================================================
def build_ngram_models(tokens):
    """Build unigram and bigram models."""
    unigrams = Counter(tokens)
    bigrams = Counter(list(ngrams(tokens, 2)))
    return unigrams, bigrams


def calculate_perplexity(bigram_model, unigram_model, test_tokens, vocab_size):
    """Calculate perplexity using Laplace smoothing."""
    N = len(test_tokens)
    log_prob = 0
    for bg in ngrams(test_tokens, 2):
        w1 = bg[0]
        count_bg = bigram_model[bg]
        count_uni = unigram_model[w1]
        prob = (count_bg + 1) / (count_uni + vocab_size)
        log_prob += -math.log(prob)
    return math.exp(log_prob / N)


# =====================================================
# 🏷️ POS TAGGING
# =====================================================
def pos_tag_text(doc):
    """Returns list of (token, POS) tuples."""
    return [(token.text, token.pos_) for token in doc if not token.is_punct and not token.is_space]


# =====================================================
# 🌳 DEPENDENCY VISUALIZATION
# =====================================================
def save_dependency_visualization(nlp_en, nlp_es, example_en, example_es, output_path="src/dependency.html"):
    """Generates dependency visualizations for English & Spanish examples and saves to HTML."""
    doc_en = nlp_en(example_en)
    doc_es = nlp_es(example_es)

    html = displacy.render(doc_en, style="dep", jupyter=False)
    html += displacy.render(doc_es, style="dep", jupyter=False)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Dependency visualizations saved to {output_path}")
