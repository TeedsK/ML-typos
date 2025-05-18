# backend/app.py
import os, time, logging
from flask import Flask, request, jsonify
from flask_cors import CORS

import model_loader

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
app.logger.setLevel(logging.DEBUG)

# --- load model once ---
with app.app_context():
    ok = model_loader.load_model()
    app.logger.info("Model init: %s", "success" if ok else "FAILED")

# ------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health_check():
    status = model_loader.symspell_model is not None
    return jsonify({
        "status": "healthy" if status else "unhealthy",
        "model_loaded": status,
        "model_name": model_loader.MODEL_NAME
    }), 200 if status else 500

# ------------------------------------------------------------------
@app.route("/api/check_typos", methods=["POST"])
def check_typos_endpoint():
    if model_loader.symspell_model is None:
        return jsonify({"error": "Model unavailable."}), 503

    data = request.get_json(silent=True) or {}
    sentence = data.get("sentence", "")
    top_k     = int(data.get("top_k", 3))

    if not isinstance(sentence, str):
        return jsonify({"error": "'sentence' must be a string"}), 400
    if not sentence.strip():
        return jsonify({
            "original_sentence": sentence,
            "corrected_sentence": "",
            "token_details": [],
            "model_name": model_loader.MODEL_NAME,
            "processing_time_ms": 0.0,
            "corrections_made": False,
            "message": "Input sentence was empty."
        }), 200

    corrected, ms, changed, token_info = model_loader.correct_text(sentence, top_k)

    return jsonify({
        "original_sentence":   sentence,
        "corrected_sentence":  corrected,
        "token_details":       token_info,      # NEW: per-token probs
        "model_name":          model_loader.MODEL_NAME,
        "processing_time_ms":  round(ms, 2),
        "corrections_made":    changed,
        "message":             "Typos checked successfully."
    }), 200

# ------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
