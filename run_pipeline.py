"""
run_pipeline.py
---------------
Runs the full multilingual NLP pipeline for English and Spanish movie reviews.
"""

import pandas as pd
import spacy
from tqdm import tqdm
from src.nlp_utils import (
    clean_text,
    tokenize_text,
    build_ngram_models,
    calculate_perplexity,
    pos_tag_text,
    save_dependency_visualization,
)

# Enable tqdm progress bar for pandas
tqdm.pandas()

# =====================================================
# 1️⃣ Load Raw Data
# =====================================================
print("📥 Loading raw datasets...")
eng = pd.read_csv("data/raw/sampled_imdb_en.csv")
spa = pd.read_csv("data/raw/sampled_imdb_es.csv")
eng["language"] = "english"
spa["language"] = "spanish"

# =====================================================
# 2️⃣ Clean Text
# =====================================================
print("🧹 Cleaning datasets...")
eng["clean_text"] = eng["review"].apply(clean_text)
spa["clean_text"] = spa["review_es"].apply(clean_text)

eng[["clean_text", "sentiment"]].to_csv("outputs/TEST_cleaned_imdb_en.csv", index=False)
spa[["clean_text", "sentiment"]].to_csv("outputs/TEST_cleaned_imdb_es.csv", index=False)
print("💾 Cleaned datasets saved successfully.")

# =====================================================
# 3️⃣ Load spaCy Models
# =====================================================
print("⚙️ Loading spaCy language models...")
nlp_en = spacy.load("en_core_web_sm")
nlp_es = spacy.load("es_core_news_sm")

# =====================================================
# 4️⃣ Tokenization
# =====================================================
print("✂️ Tokenizing reviews... (this may take several minutes)")

eng["tokens"] = eng["clean_text"].progress_apply(lambda x: tokenize_text(nlp_en(x)))
spa["tokens"] = spa["clean_text"].progress_apply(lambda x: tokenize_text(nlp_es(x)))

print("✅ Tokenization complete for both datasets.\n")

# =====================================================
# 5️⃣ N-gram Models + Perplexity
# =====================================================
print("📊 Building N-gram models & calculating perplexity...")
tokens_en = [t for tokens in eng["tokens"] for t in tokens]
tokens_es = [t for tokens in spa["tokens"] for t in tokens]

split_en = int(0.95 * len(tokens_en))
split_es = int(0.95 * len(tokens_es))
train_en, test_en = tokens_en[:split_en], tokens_en[split_en:]
train_es, test_es = tokens_es[:split_es], tokens_es[split_es:]

model_en_uni, model_en_bi = build_ngram_models(train_en)
model_es_uni, model_es_bi = build_ngram_models(train_es)

pp_en = calculate_perplexity(model_en_bi, model_en_uni, test_en, len(model_en_uni))
pp_es = calculate_perplexity(model_es_bi, model_es_uni, test_es, len(model_es_uni))

print(f"✅ English Perplexity: {pp_en:.2f}")
print(f"✅ Spanish Perplexity: {pp_es:.2f}")

# =====================================================
# 6️⃣ POS Tagging
# =====================================================
print("🏷️ Applying POS tagging... (this may take several minutes)")

eng["pos_tags"] = eng["clean_text"].progress_apply(lambda x: pos_tag_text(nlp_en(x)))
spa["pos_tags"] = spa["clean_text"].progress_apply(lambda x: pos_tag_text(nlp_es(x)))

print("✅ POS tagging complete for both datasets.\n")

# =====================================================
# 8️⃣ Save Final Outputs
# =====================================================
eng.to_csv("outputs/TEST_tokenized_pos_en.csv", index=False)
spa.to_csv("outputs/TEST_tokenized_pos_es.csv", index=False)
print("\n🎉 Full NLP pipeline executed successfully!")
