import sys
import os
import time
import threading
import webbrowser

# Make src/ importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, render_template, jsonify
import pandas as pd
import spacy

from src.nlp_utils import (
    clean_text,
    tokenize_text,
    pos_tag_text,
    extract_entities,
    train_sentiment_classifier,
    evaluate_classifier,
    predict_sentiment
)

app = Flask(__name__)

############################################################
# GLOBAL STATE
############################################################

STATE = {
    "running": False,
    "progress": 0,
    "status": "Idle",
    "count_en": 0,
    "count_es": 0,
    "total_en": 0,
    "total_es": 0
}


STOP_REQUESTED = False
STOP_CONFIRMED = False
EVAL_RESULTS = None


############################################################
# STATE HELPERS
############################################################

def set_state(progress=None, status=None, running=None,count_en=None, count_es=None):
    if progress is not None:
        STATE["progress"] = progress
    if status is not None:
        STATE["status"] = status
    if running is not None:
        STATE["running"] = running
    if count_en is not None:
        STATE["count_en"] = count_en
    if count_es is not None:
        STATE["count_es"] = count_es



def reset_stop():
    global STOP_REQUESTED, STOP_CONFIRMED
    STOP_REQUESTED = False
    STOP_CONFIRMED = False


def request_stop():
    global STOP_REQUESTED
    STOP_REQUESTED = True
    STATE["status"] = "Stopping..."


def check_stop():
    """Call this inside loops — immediately stops execution."""
    if STOP_REQUESTED:
        raise KeyboardInterrupt("Pipeline stop requested.")


############################################################
# PIPELINE STEPS
############################################################

def run_pipeline():
    global EVAL_RESULTS
    reset_stop()
    EVAL_RESULTS = None
    set_state(running=True, progress=0, status="Loading datasets...")

    try:
        ############################################################
        # 1) LOAD DATA
        ############################################################
        eng = pd.read_csv("data/raw/sampled_imdb_en.csv")
        spa = pd.read_csv("data/raw/sampled_imdb_es.csv")
        time.sleep(0.3)
        check_stop()

        # Initialize dataset counters
        STATE["total_en"] = len(eng)
        STATE["total_es"] = len(spa)
        STATE["count_en"] = 0
        STATE["count_es"] = 0

        ############################################################
        # 2) CLEANING
        ############################################################
        set_state(progress=10, status="Cleaning text...")

        eng["clean_text"] = eng["review"].apply(clean_text)
        spa["clean_text"] = spa["review_es"].apply(clean_text)

        eng.to_csv("outputs/STEP1_clean_en.csv", index=False)
        spa.to_csv("outputs/STEP1_clean_es.csv", index=False)

        time.sleep(0.3)
        check_stop()

        ############################################################
        # 3) LOAD SPACY MODELS
        ############################################################
        set_state(progress=20, status="Loading language models...")

        nlp_en = spacy.load("en_core_web_sm")
        nlp_es = spacy.load("es_core_news_sm")

        time.sleep(0.2)
        check_stop()

        ############################################################
        # 4) TOKENIZATION
        ############################################################
        set_state(progress=35, status="Tokenizing...")

        eng_tokens = []
        for i, txt in enumerate(eng["clean_text"], start=1):
            check_stop()
            eng_tokens.append(tokenize_text(nlp_en(txt)))
            set_state(count_en=i)
        eng["tokens"] = eng_tokens

        spa_tokens = []
        for i, txt in enumerate(spa["clean_text"], start=1):
            check_stop()
            spa_tokens.append(tokenize_text(nlp_es(txt)))
            set_state(count_es=i)
        spa["tokens"] = spa_tokens

        eng["tokens"].to_csv("outputs/STEP2_tokens_en.csv", index=False)
        spa["tokens"].to_csv("outputs/STEP2_tokens_es.csv", index=False)

        time.sleep(0.2)
        check_stop()

        ############################################################
        # 5) POS TAGGING
        ############################################################
        set_state(progress=50, status="POS tagging...")

        STATE["count_en"] = 0
        STATE["count_es"] = 0

        eng_pos = []
        for i, txt in enumerate(eng["clean_text"], start=1):
            check_stop()
            eng_pos.append(pos_tag_text(nlp_en(txt)))
            set_state(count_en=i)
        eng["pos_tags"] = eng_pos

        spa_pos = []
        for i, txt in enumerate(spa["clean_text"], start=1):
            check_stop()
            spa_pos.append(pos_tag_text(nlp_es(txt)))
            set_state(count_es=i)
        spa["pos_tags"] = spa_pos

        eng["pos_tags"].to_csv("outputs/STEP3_pos_en.csv", index=False)
        spa["pos_tags"].to_csv("outputs/STEP3_pos_es.csv", index=False)

        time.sleep(0.2)
        check_stop()

        ############################################################
        # 6) NAMED ENTITY RECOGNITION (SAVE SEPARATELY)
        ############################################################
        set_state(progress=65, status="Extracting named entities...")


        STATE["count_en"] = 0
        STATE["count_es"] = 0

        eng_ents = []
        for i, txt in enumerate(eng["clean_text"], start=1):
            check_stop()
            eng_ents.append(extract_entities(nlp_en(txt)))
            set_state(count_en=i)
        eng["entities"] = eng_ents

        spa_ents = []
        for i, txt in enumerate(spa["clean_text"], start=1):
            check_stop()
            spa_ents.append(extract_entities(nlp_es(txt)))
            set_state(count_es=i)
        spa["entities"] = spa_ents

        eng["entities"].to_csv("outputs/STEP4_ner_en.csv", index=False)
        spa["entities"].to_csv("outputs/STEP4_ner_es.csv", index=False)

        time.sleep(0.2)
        check_stop()

        ############################################################
        # 7) SENTIMENT CLASSIFICATION + EVALUATION
        ############################################################
        set_state(progress=80, status="Sentiment classification...")


        STATE["count_en"] = 0
        STATE["count_es"] = 0

        # ENGLISH
        model_en, vec_en, _ = train_sentiment_classifier(
            eng["clean_text"], eng["sentiment"]
        )
        eng_pred = []
        for i, txt in enumerate(eng["clean_text"], start=1):
            check_stop()
            eng_pred.append(predict_sentiment(model_en, vec_en, txt))
            set_state(count_en=i)
        eng["predicted_sentiment"] = eng_pred

        report_en, acc_en = evaluate_classifier(
            model_en, vec_en, eng["clean_text"], eng["sentiment"]
        )
        check_stop()

        # SPANISH
        model_es, vec_es, _ = train_sentiment_classifier(
            spa["clean_text"], spa["sentiment"]
        )
        spa_pred = []
        for i, txt in enumerate(spa["clean_text"], start=1):
            check_stop()
            spa_pred.append(predict_sentiment(model_es, vec_es, txt))
            set_state(count_es=i)
        spa["predicted_sentiment"] = spa_pred

        report_es, acc_es = evaluate_classifier(
            model_es, vec_es, spa["clean_text"], spa["sentiment"]
        )
        check_stop()

        # Store evaluation
        EVAL_RESULTS = {
            "english": {"accuracy": acc_en, "report": report_en},
            "spanish": {"accuracy": acc_es, "report": report_es},
        }

                # Convert reports into dictionaries for UI tables
        from sklearn.metrics import classification_report

        # Clean English report into dictionary
        report_dict_en = classification_report(
            eng["sentiment"], eng["predicted_sentiment"], output_dict=True
        )

        # Clean Spanish report into dictionary
        report_dict_es = classification_report(
            spa["sentiment"], spa["predicted_sentiment"], output_dict=True
        )

        # Save full text reports
        with open("outputs/REPORT_en.txt", "w") as f:
            f.write(report_en)

        with open("outputs/REPORT_es.txt", "w") as f:
            f.write(report_es)

        # Save metrics table as CSV
        pd.DataFrame(report_dict_en).transpose().to_csv("outputs/REPORT_en_metrics.csv")
        pd.DataFrame(report_dict_es).transpose().to_csv("outputs/REPORT_es_metrics.csv")

        # Store everything for the UI
        EVAL_RESULTS = {
            "english": {
                "accuracy": acc_en,
                "report": report_en,
                "metrics": report_dict_en
            },
            "spanish": {
                "accuracy": acc_es,
                "report": report_es,
                "metrics": report_dict_es
            }
        }


        ############################################################
        # 8) SAVE FINAL OUTPUTS (CLEAN + REAL SENTIMENT + PREDICTED)
        ############################################################
        set_state(progress=95, status="Saving final outputs...")

        eng[["clean_text", "sentiment", "predicted_sentiment"]].to_csv(
            "outputs/FINAL_sentiment_predictions_en.csv", index=False
        )
        spa[["clean_text", "sentiment", "predicted_sentiment"]].to_csv(
            "outputs/FINAL_sentiment_predictions_es.csv", index=False
        )

        ############################################################
        # DONE
        ############################################################
        set_state(progress=100, status="Pipeline completed ✓", running=False)

    except KeyboardInterrupt:
        set_state(status="Stopped by user", running=False)
        reset_stop()

    except Exception as e:
        set_state(status=f"Error: {str(e)}", running=False)
        reset_stop()

############################################################
# FLASK ROUTES
############################################################

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start():
    if STATE["running"]:
        return jsonify({"status": "already_running"})

    t = threading.Thread(target=run_pipeline, daemon=True)
    t.start()
    return jsonify({"status": "started"})


@app.route("/stop", methods=["POST"])
def stop():
    if not STATE["running"]:
        return jsonify({"status": "not_running"})
    request_stop()
    return jsonify({"status": "stopping"})


@app.route("/progress")
def progress():
    return jsonify(STATE)


@app.route("/evaluation")
def evaluation():
    global EVAL_RESULTS
    if EVAL_RESULTS is None:
        return jsonify({"error": "Evaluation not ready yet."})
    return jsonify(EVAL_RESULTS)


@app.route("/view/<step>")
def view(step):
    """Return first 50 rows of selected CSV."""
    file_map = {
        "clean_en": "outputs/STEP1_clean_en.csv",
        "clean_es": "outputs/STEP1_clean_es.csv",

        "tokens_en": "outputs/STEP2_tokens_en.csv",
        "tokens_es": "outputs/STEP2_tokens_es.csv",

        "pos_en": "outputs/STEP3_pos_en.csv",
        "pos_es": "outputs/STEP3_pos_es.csv",

        "ner_en": "outputs/STEP4_ner_en.csv",
        "ner_es": "outputs/STEP4_ner_es.csv",

        "final_en": "outputs/FINAL_sentiment_predictions_en.csv",
        "final_es": "outputs/FINAL_sentiment_predictions_es.csv",
    }

    path = file_map.get(step)
    if path is None or not os.path.exists(path):
        return jsonify({"error": "file not available"}), 404

    df = pd.read_csv(path)
    return df.head(50).to_json(orient="records", force_ascii=False)


############################################################
# AUTO-OPEN BROWSER + RUN SERVER
############################################################

def open_browser():
    time.sleep(1)
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    threading.Thread(target=open_browser).start()
    app.run(debug=True,use_reloader=False)
