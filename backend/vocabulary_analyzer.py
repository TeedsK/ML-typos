import pandas as pd
import re
from collections import Counter
import logging
import os

# --- Configuration ---
DATASET_DIRECTORY = "."
DATASET_NAME = "developer_typo_dataset_1000.csv"
LOG_FILE = "vocabulary_analysis_log.txt"
TOP_N_WORDS = 50 # How many of the most common words to display

# --- Logger Setup ---
def setup_logger(log_file):
    logger = logging.getLogger("VocabularyAnalyzer")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.DEBUG)
        formatter_fh = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter_fh)
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter_ch = logging.Formatter('%(levelname)s: %(message)s')
        ch.setFormatter(formatter_ch)
        logger.addHandler(ch)
    return logger

logger = setup_logger(LOG_FILE)

# --- Re-use the improved_tokenizer from model_trainer.py ---
def improved_tokenizer(text):
    if not isinstance(text, str):
        logger.warning(f"Expected string for tokenization, got {type(text)}. Returning empty list.")
        return []
    text = text.lower()
    tokens = re.findall(r'\b[\w/:.#+-]+\b|[.,!?;()\'":]', text)
    return [token for token in tokens if token]

# --- Main Script ---
if __name__ == "__main__":
    logger.info("--- Starting Vocabulary Analysis ---")
    data_file_path = os.path.join(DATASET_DIRECTORY, DATASET_NAME)
    logger.info(f"Loading dataset from: {data_file_path}")

    try:
        df = pd.read_csv(data_file_path)
        logger.info(f"Dataset '{DATASET_NAME}' loaded successfully with {len(df)} rows.")
    except FileNotFoundError:
        logger.error(f"Dataset file not found at {data_file_path}.")
        exit()
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        exit()

    if 'corrected_text' not in df.columns:
        logger.error("Dataset must contain 'corrected_text' column.")
        exit()

    logger.info("Analyzing vocabulary of 'corrected_text' column...")
    all_tokens = []
    # Let's also log tokens for the first few sentences to see them directly
    logger.info("--- Tokens from the first 5 'corrected_text' sentences ---")
    for i, text in enumerate(df['corrected_text'].head(5)):
        tokens = improved_tokenizer(text)
        logger.info(f"Sentence {i+1} tokens: {tokens}")
        all_tokens.extend(tokens)
    
    # Process the rest of the sentences
    for text in df['corrected_text'][5:]:
        tokens = improved_tokenizer(text)
        all_tokens.extend(tokens)
        
    # Filter out tokens that are just punctuation for the frequency count,
    # as these are not part of the spell-checking dictionary in the same way.
    # The SymSpell dictionary itself *will* contain them if they are frequent,
    # but for understanding word vocabulary, we exclude them here.
    punctuation_to_exclude_from_counts = ['.', ',', '!', '?', ';', '(', ')', '\'', '"', ':']
    words_only = [token for token in all_tokens if token not in punctuation_to_exclude_from_counts]

    word_counts = Counter(words_only)

    logger.info(f"\nTotal number of tokens (including punctuation) found in 'corrected_text': {len(all_tokens)}")
    logger.info(f"Number of unique tokens (words/terms, excluding standalone punctuation listed) in 'corrected_text': {len(word_counts)}")
    
    logger.info(f"\n--- Top {TOP_N_WORDS} most common words/terms (and their counts) ---")
    for i, (word, count) in enumerate(word_counts.most_common(TOP_N_WORDS)):
        logger.info(f"{i+1}. '{word}': {count}")

    # Check if all words in the generated SymSpell dictionary are found by this script's tokenizer
    symspell_dict_path = os.path.join(DATASET_DIRECTORY, "developer_symspell_dict_revised.txt")
    try:
        symspell_dict_terms = set()
        with open(symspell_dict_path, "r", encoding="utf-8") as f:
            for line in f:
                term = line.split(" ")[0]
                symspell_dict_terms.add(term)
        
        logger.info(f"\nNumber of terms in loaded SymSpell dictionary file: {len(symspell_dict_terms)}")
        
        missing_from_vocab_analysis = symspell_dict_terms - set(word_counts.keys())
        if missing_from_vocab_analysis:
            logger.warning(f"The following {len(missing_from_vocab_analysis)} terms were in the SymSpell dictionary but NOT found by this vocabulary analysis script's tokenization of 'corrected_text'. This might indicate an issue!")
            logger.warning(f"Missing terms: {missing_from_vocab_analysis}")
        else:
            logger.info("All terms from the SymSpell dictionary file are accounted for by this script's vocabulary analysis. Good consistency.")

    except FileNotFoundError:
        logger.warning(f"SymSpell dictionary file not found at {symspell_dict_path}. Cannot perform consistency check.")


    logger.info("\n--- Vocabulary Analysis Completed ---")
    logger.info(f"Log file generated at: {os.path.abspath(LOG_FILE)}")