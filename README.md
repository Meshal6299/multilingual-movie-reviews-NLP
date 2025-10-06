# 🎬 Multilingual Movie Reviews NLP Pipeline

This project builds an end-to-end **Natural Language Processing (NLP) pipeline** to analyze movie reviews in **English** and **Spanish**, performing:
- Text Cleaning & Tokenization  
- POS Tagging & Parsing  
- Named Entity Recognition (NER)  
- Sentiment Analysis (Positive/Negative Classification)

The project is modular, fully reproducible, and runs on a small sampled dataset (2K reviews total).

---

## 🧠 Project Overview

| Component | Description |
|------------|--------------|
| **Languages** | English 🇬🇧 & Spanish 🇪🇸 |
| **Dataset** | 1K English + 1K Spanish movie reviews (balanced positive/negative) |
| **Goal** | Detect sentiment and extract named entities |
| **Libraries Used** | `pandas`, `spaCy`, `nltk`, `scikit-learn`, `matplotlib`, `seaborn` |
| **Pipeline Entry** | `run_pipeline.py` |
| **Environment** | Python Virtual Environment (`.venv`) |

---

## 🧩 Folder Structure

```
├── .venv/ # Virtual environment (local only)
├── data/
│   ├── processed/
│   └── raw/
│       ├── sampleChecker.py # checker for even distribution of positives and negatives
│       ├── sampled_imdb_en.csv # English dataset
│       └── sampled_imdb_es.csv # Spanish dataset
├── notebooks/
│   ├── sampleGenerator_en.ipynb # English dataset generation
│   └── sampleGenerator_es.ipynb # Spanish dataset generation
├── src/
├── .gitignore
├── README.md
├── requirements.txt # Dependencies
└── run_pipeline.py # Main enrty point
```

