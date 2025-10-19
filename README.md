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
---

## 📚 Dataset Information

The original datasets are sourced from **Kaggle** and have been legally sampled for academic use:

1. **English Dataset:**  
   *IMDB Dataset of 50K Movie Reviews* — includes labeled positive and negative reviews.  
   🔗 https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data

2. **Spanish Dataset:**  
   *IMDB Dataset of 50K Movie Reviews (Spanish Translation)* — machine-translated and labeled for sentiment.  
   🔗 https://www.kaggle.com/datasets/luisdiegofv97/imdb-dataset-of-50k-movie-reviews-spanish/data

Each dataset was reduced to **1,000 randomly sampled reviews per language** to ensure balanced sentiment distribution and faster model training.
