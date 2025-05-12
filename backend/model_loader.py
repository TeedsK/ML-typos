# backend/model_loader.py (Version 1.1 with minor rejoin tweak)
import os
import re
import time 
from symspellpy import SymSpell, Verbosity
import logging

# --- Configuration ---
DICTIONARY_DIR = "." 
DICTIONARY_FILENAME = "developer_symspell_dict_train_only.txt" # From Step 4
SYMSPELL_DICTIONARY_PATH = os.path.join(DICTIONARY_DIR, DICTIONARY_FILENAME)

MODEL_NAME = "SymSpell_DevTypo_v1.1" # Updated model name
MAX_EDIT_DISTANCE_LOOKUP = 2
PREFIX_LENGTH = 7

logger = logging.getLogger("SymSpellModel") 

# --- Text Preprocessing Functions (Using V2) ---
def improved_tokenizer_v2(text):
    if not isinstance(text, str): return []
    text = text.lower()
    # Ensure no empty strings are produced by findall if regex matches empty parts
    tokens = [token for token in re.findall(r'\b[\w/:.#+-]+\b|%|[.,!?;()\'":]', text) if token]
    return tokens

def rejoin_tokens_v2(tokens):
    if not tokens: return ""
    
    # Filter out any leading empty strings or None values from tokens list if they sneak in
    # This helps prevent a leading space if the first "corrected" token was empty
    processed_tokens = [token for token in tokens if token is not None and token != '']
    if not processed_tokens: return ""

    # Start with the first valid token directly
    sentence = processed_tokens[0] 
    for i in range(1, len(processed_tokens)):
        token = processed_tokens[i]
        # Add space before token unless it's specific punctuation.
        if token in ['.', ',', '!', '?', ';', ':', ')', '\'', '"'] or \
           (token == '%' and (processed_tokens[i-1].isdigit() or processed_tokens[i-1].endswith("d"))):
            pass # No space for these cases
        else:
            sentence += " " # Add a space before other tokens
        sentence += token
    
    # Correct space after opening parenthesis (if any are standalone tokens)
    sentence = re.sub(r'([(])\s+', r'\1', sentence) 
    return sentence.strip()

# --- Global SymSpell Model Instance ---
symspell_model = None

def load_model():
    """Loads the SymSpell model into the global symspell_model variable."""
    global symspell_model
    if symspell_model is not None:
        logger.info("SymSpell model already loaded.")
        return True

    logger.info("Initializing SymSpell model for API.")
    symspell_instance = SymSpell(max_dictionary_edit_distance=MAX_EDIT_DISTANCE_LOOKUP, prefix_length=PREFIX_LENGTH)
    
    if not os.path.exists(SYMSPELL_DICTIONARY_PATH):
        logger.error(f"SymSpell dictionary file not found at {SYMSPELL_DICTIONARY_PATH}. Model cannot be loaded.")
        return False

    logger.info(f"Loading dictionary from {SYMSPELL_DICTIONARY_PATH} into SymSpell model.")
    if symspell_instance.load_dictionary(SYMSPELL_DICTIONARY_PATH, term_index=0, count_index=1, encoding="utf-8"):
        symspell_model = symspell_instance
        logger.info("SymSpell model loaded successfully for API.")
        return True
    else:
        logger.error("Failed to load SymSpell dictionary for API.")
        return False

def correct_text(sentence_to_correct):
    """Corrects a sentence using the loaded SymSpell model."""
    global symspell_model
    if not symspell_model:
        logger.error("SymSpell model not loaded. Cannot correct sentence.")
        # Return original sentence and indicate no correction if model is not loaded
        return sentence_to_correct, 0.0, False 

    start_time = time.time()
    
    # Tokenize the input sentence (original, before any correction attempt)
    original_lower_tokens = improved_tokenizer_v2(sentence_to_correct)
    
    # Perform correction
    corrected_sentence_text = correct_sentence_symspell_internal(sentence_to_correct)
    
    processing_time_ms = (time.time() - start_time) * 1000

    # Tokenize the corrected sentence to check if changes were made
    corrected_lower_tokens = improved_tokenizer_v2(corrected_sentence_text)
    
    # Determine if corrections were made
    corrections_were_made = original_lower_tokens != corrected_lower_tokens
    
    return corrected_sentence_text, processing_time_ms, corrections_were_made

def correct_sentence_symspell_internal(sentence):
    """Internal logic for SymSpell correction, assumes model is loaded."""
    global symspell_model
    
    original_tokens = improved_tokenizer_v2(sentence) # Lowercases and tokenizes input
    if not original_tokens: # Handle cases where tokenization results in empty list (e.g. empty or only whitespace input)
        return ""
        
    corrected_tokens = []
    keep_as_is_tokens = ['.', ',', '!', '?', ';', '(', ')', '\'', '"', ':', '%']

    for token in original_tokens:
        if token in keep_as_is_tokens:
            corrected_tokens.append(token)
            continue
        
        suggestions = symspell_model.lookup(token, Verbosity.CLOSEST, 
                                           MAX_EDIT_DISTANCE_LOOKUP, 
                                           include_unknown=True, 
                                           transfer_casing=False) 
        if suggestions:
            corrected_tokens.append(suggestions[0].term)
        else:
            # This path should ideally not be taken frequently if include_unknown=True,
            # as it should return the original token itself if no suggestion is found.
            # If symspell.lookup returns an empty list (which is unusual for include_unknown=True),
            # append the original token to avoid losing it.
            corrected_tokens.append(token) 
            
    return rejoin_tokens_v2(corrected_tokens)