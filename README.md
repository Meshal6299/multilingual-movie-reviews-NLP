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
├── data
│   ├── processed
│   └── raw
│       ├── sampleChecker.py
│       ├── sampled_imdb_en.csv
│       └── sampled_imdb_es.csv
├── notebooks
│   ├── sampleGenerator_en.ipynb
│   └── sampleGenerator_es.ipynb
├── README.md
├── requirements.txt
├── run_pipeline.py
└── src
```

