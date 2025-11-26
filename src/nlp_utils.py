"""
src/nlp_utils.py
----------------
Utility functions for multilingual NLP pipeline:
- Text cleaning
- Tokenization
- N-gram modeling & perplexity
- POS tagging
- Dependency parsing visualization
- Named Entity Recognition (NER)
- Sentiment Classification (TF-IDF + Logistic Regression)
"""

import re
import math
import contractions
import spacy
from nltk import ngrams
from collections import Counter
from spacy import displacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


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


# =====================================================
# 🧠 NAMED ENTITY RECOGNITION
# =====================================================
def extract_entities(doc):
    """Extracts named entities as (text, label) tuples."""
    return [(ent.text, ent.label_) for ent in doc.ents]


# =====================================================
# ❤️ SENTIMENT CLASSIFICATION
# =====================================================
def train_sentiment_classifier(texts, labels):
    """
    Trains a Logistic Regression sentiment classifier using TF-IDF features.
    Returns model, vectorizer, and classification report.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=42, stratify=labels
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=300)
    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)
    report = classification_report(y_test, predictions)

    return model, vectorizer, report


def predict_sentiment(model, vectorizer, text: str):
    """Predicts sentiment label for a single text input."""
    x = vectorizer.transform([text])
    return model.predict(x)[0]


# =====================================================
# 📊 CLASSIFIER EVALUATION
# =====================================================
def evaluate_classifier(model, vectorizer, texts, labels):
    """
    Evaluates the trained sentiment classifier using TF-IDF features.
    Returns:
        - classification report (string)
        - accuracy (float)
    """
    # Transform all texts using the fitted vectorizer
    X_vec = vectorizer.transform(texts)

    # Predict all labels
    predictions = model.predict(X_vec)

    # Generate full classification report
    report_str = classification_report(labels, predictions, output_dict=False)

    # Dict version to extract accuracy
    report_dict = classification_report(labels, predictions, output_dict=True)
    accuracy = report_dict.get("accuracy", 0.0)

    return report_str, accuracy

