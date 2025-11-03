# 🎬 Multilingual Movie Reviews NLP Pipeline

This project builds an end-to-end **Natural Language Processing (NLP) pipeline** to analyze movie reviews in **English** and **Spanish**, performing:
- Text Cleaning & Tokenization  
- POS Tagging & Parsing  
- Named Entity Recognition (NER)  
- Sentiment Analysis (Positive/Negative Classification)

The project is modular, fully reproducible, and runs on a small sampled dataset (20K reviews total).

---

## 🧠 Project Overview

| Component | Description |
|------------|--------------|
| **Languages** | English 🇬🇧 & Spanish 🇪🇸 |
| **Dataset** | 10K English + 10K Spanish movie reviews (balanced positive/negative) |
| **Goal** | Implement end-to-end NLP pipeline, Detect sentiment and extract named entities |
| **Libraries Used** | `pandas`, `spaCy`, `nltk`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`, `contractions`, `math`, `re`, `ngrams`, `displacy`, `counter` |
| **Pipeline Entry** | `run_pipeline.py` |
| **Environment** | Python Virtual Environment (`.venv`) |

---

## 🧩 Folder Structure

```
├── .venv/
├── data/
│   ├── processed/
│   │   ├── 01_cleaned_imdb_en.csv
│   │   ├── 01_cleaned_imdb_es.csv
│   │   ├── 02_tokenized_pos_imdb_en.csv
│   │   └── 02_tokenized_pos_imdb_es.csv
│   └── raw/
│       ├── MUSTREAD.txt
│       ├── sampled_imdb_en.csv
│       └── sampled_imdb_es.csv
├── notebooks/
│   ├── 01_data_cleaning_eda.ipynb
│   ├── 02_tokenization_ngram_pos.ipynb
│   ├── sampleChecker.ipynb
│   ├── sampleGenerator_en.ipynb
│   └── sampleGenerator_es.ipynb
├── outputs/
│   ├── TEST_cleaned_imdb_en.csv
│   ├── TEST_cleaned_imdb_es.csv
│   ├── TEST_tokenized_pos_en.csv
│   └── TEST_tokenized_pos_es.csv
├── src/
│   ├── dependency.html
│   └── nlp_utils.py
├── .gitignore
├── README.md
├── requirements.txt
└── run_pipeline.py
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

Each dataset was reduced to **10,000 randomly sampled reviews per language** to ensure balanced sentiment distribution and faster model training.


## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Meshal6299/multilingual-movie-reviews-NLP.git

cd multilingual-movie-reviews-NLP
```

### 2️⃣ Create and Activate a Virtual Environment
```bash
python -m venv .venv

# 🪟 Windows
.venv\Scripts\activate
# 🐧 macOS / Linux
source .venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Download spaCy Models
```bash
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
```

### 5️⃣ Run the Full NLP Pipeline  
```bash
python run_pipeline.py
```

**This will automatically:**
1. Clean and normalize raw data 
2. Tokenize English & Spanish reviews
3. Build N-gram models and calculate perplexity
4. Apply POS tagging

All outputs are saved under `outputs/`.
