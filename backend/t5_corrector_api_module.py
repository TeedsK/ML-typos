# backend/t5_corrector_api_module.py
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
import time
import logging
import os

# --- Configuration ---
MODEL_PATH = "./t5_typo_corrector_v1"  # Path where your fine-tuned model is saved
MODEL_NAME_FOR_API = "T5-small_DevTypo_v1_95k" # Name to show in API response
TASK_PREFIX = "fix typos: "

# Logger
logger = logging.getLogger("T5Corrector")
if not logger.handlers: # Avoid adding multiple handlers if re-imported
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# --- Global Model and Tokenizer Instances ---
t5_model = None
t5_tokenizer = None
device = None

def load_t5_model_and_tokenizer():
    """Loads the fine-tuned T5 model and tokenizer."""
    global t5_model, t5_tokenizer, device

    if t5_model is not None and t5_tokenizer is not None:
        logger.info("T5 model and tokenizer already loaded.")
        return True

    if not os.path.exists(MODEL_PATH):
        logger.error(f"Fine-tuned T5 model not found at {MODEL_PATH}. Cannot load.")
        return False
    
    logger.info(f"Loading fine-tuned T5 model from {MODEL_PATH}...")
    try:
        # Determine device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        # elif torch.backends.mps.is_available() and torch.backends.mps.is_built(): # For Apple Silicon
        #     device = torch.device("mps")
        #     logger.info("Using Apple MPS.")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU for T5 model (inference might be slow).")

        t5_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, legacy=False)
        t5_model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
        t5_model.to(device) # Move model to the determined device
        t5_model.eval() # Set model to evaluation mode
        logger.info("Fine-tuned T5 model and tokenizer loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Error loading T5 model/tokenizer: {e}", exc_info=True)
        t5_model = None
        t5_tokenizer = None
        return False

def correct_text_t5(sentence_to_correct: str, max_length: int = 128) -> tuple[str | None, float, bool]:
    """
    Corrects a sentence using the loaded fine-tuned T5 model.
    Returns (corrected_sentence, processing_time_ms, corrections_made)
    """
    global t5_model, t5_tokenizer, device

    if not t5_model or not t5_tokenizer:
        logger.error("T5 model or tokenizer not loaded. Cannot correct sentence.")
        return sentence_to_correct, 0.0, False # Return original, no time, no corrections

    start_time = time.time()
    
    input_text = TASK_PREFIX + sentence_to_correct
    
    try:
        inputs = t5_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        
        # Generate prediction
        # You can adjust generation parameters like num_beams, do_sample, top_k, top_p for different results
        outputs = t5_model.generate(
            inputs.input_ids, 
            attention_mask=inputs.attention_mask,
            max_length=max_length + 20, # Allow for slightly longer output
            num_beams=4, # Beam search can often produce better results
            early_stopping=True
        )
        
        corrected_sentence = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Check if corrections were made (simple string comparison after lowercasing and stripping)
        # More robust would be token-level, but for T5 output this is often sufficient
        corrections_made = sentence_to_correct.strip().lower() != corrected_sentence.strip().lower()

        return corrected_sentence, processing_time_ms, corrections_made
    except Exception as e:
        logger.error(f"Error during T5 prediction: {e}", exc_info=True)
        return sentence_to_correct, (time.time() - start_time) * 1000, False


if __name__ == '__main__':
    # Quick test
    if load_t5_model_and_tokenizer():
        test_sentences = [
            "a machine learning and fuil stack engineer huilding web tools and apps that deliver measurabie impacts",
            "this is a testt sentance with some errrors.",
            "what hapens if i give it corect text?",
            "how well dose it fix PH,P or COOL to COBOL"
        ]
        for sentence in test_sentences:
            corrected, proc_time, made_corrections = correct_text_t5(sentence)
            print("-" * 30)
            print(f"Original:    '{sentence}'")
            print(f"T5 Corrected:'{corrected}' (Time: {proc_time:.2f}ms, Corrections: {made_corrections})")
    else:
        print("Failed to load T5 model for testing.")