# backend/app.py
import os
from flask import Flask, request, jsonify
from flask_cors import CORS # For handling Cross-Origin Resource Sharing
import logging
import time

# Import model loading and correction functions from model_loader
import model_loader 

app = Flask(__name__)
CORS(app) # This will allow requests from your React frontend (localhost:3000 usually)

# --- Logging Setup ---
# You can customize logging further (e.g., log to a file)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
app.logger.setLevel(logging.DEBUG) # Flask's own logger

# --- Model Loading ---
# Attempt to load the model when the application starts.
# In a production environment with multiple workers (e.g., Gunicorn),
# this might need to be handled per worker or using a shared mechanism.
with app.app_context():
    if not model_loader.load_model():
        app.logger.error("CRITICAL: SymSpell model failed to load on application startup. API may not function correctly.")
    else:
        app.logger.info("SymSpell model loaded successfully on application startup.")


@app.route('/api/health', methods=['GET'])
def health_check():
    """A simple health check endpoint."""
    app.logger.info("Health check endpoint called.")
    # Check if model is loaded as part of health
    if model_loader.symspell_model is not None:
        return jsonify({"status": "healthy", "message": "Backend is running and model is loaded."}), 200
    else:
        app.logger.warning("Health check: Model is not loaded.")
        return jsonify({"status": "unhealthy", "message": "Backend is running but model is not loaded."}), 500


@app.route('/api/check_typos', methods=['POST'])
def check_typos_endpoint():
    app.logger.info(f"'{request.method}' request received for '{request.path}'")
    
    if model_loader.symspell_model is None:
        app.logger.error("Model is not loaded. Cannot process request.")
        return jsonify({"error": "Model is not available. Please try again later."}), 503 # Service Unavailable

    try:
        data = request.get_json()
        app.logger.debug(f"Request JSON data: {data}")

        if not data or 'sentence' not in data:
            app.logger.warning("Request missing 'sentence' field.")
            return jsonify({"error": "Missing 'sentence' field in request body"}), 400

        original_sentence = data['sentence']
        if not isinstance(original_sentence, str):
            app.logger.warning("'sentence' field is not a string.")
            return jsonify({"error": "'sentence' must be a string"}), 400
        
        if not original_sentence.strip():
            app.logger.info("Received empty sentence for typo checking.")
            # Return values consistent with frontend expectation even for empty input
            return jsonify({
                "original_sentence": original_sentence,
                "corrected_sentence": "", # Or original_sentence
                "model_name": model_loader.MODEL_NAME,
                "processing_time_ms": 0.0,
                "corrections_made": False,
                "message": "Input sentence was empty."
            }), 200

        app.logger.info(f"Processing sentence for typos: '{original_sentence}'")
        
        corrected_sentence, processing_time_ms, corrections_made = model_loader.correct_text(original_sentence)
        
        app.logger.info(f"Original: '{original_sentence}', Corrected: '{corrected_sentence}', Time: {processing_time_ms:.2f}ms, Corrections Made: {corrections_made}")

        response_data = {
            "original_sentence": original_sentence,
            "corrected_sentence": corrected_sentence,
            "model_name": model_loader.MODEL_NAME,
            "processing_time_ms": round(processing_time_ms, 2),
            "corrections_made": corrections_made,
            "message": "Typos checked successfully."
        }
        app.logger.debug(f"Sending response: {response_data}")
        return jsonify(response_data), 200

    except Exception as e:
        app.logger.error(f"An error occurred in /api/check_typos: {str(e)}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500


if __name__ == '__main__':
    # Make sure the dictionary file is in the correct path relative to this script
    # or update SYMSPELL_DICTIONARY_PATH in model_loader.py
    app.logger.info(f"Expected dictionary path: {os.path.abspath(model_loader.SYMSPELL_DICTIONARY_PATH)}")
    if not os.path.exists(model_loader.SYMSPELL_DICTIONARY_PATH):
         app.logger.warning(f"DICTIONARY FILE NOT FOUND AT: {model_loader.SYMSPELL_DICTIONARY_PATH}. The API will likely fail to correct text.")
    
    app.run(host='0.0.0.0', port=5001, debug=True) # debug=True is for development