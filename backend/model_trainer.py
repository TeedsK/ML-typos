import pandas as pd
import logging
import os
import re
from collections import Counter
from symspellpy import SymSpell, Verbosity
from sklearn.model_selection import train_test_split # For splitting data
# To calculate Word Error Rate (WER), you might need an external library like 'jiwer'
# We can add this later if desired. For now, let's do sentence accuracy.
# try:
#     import jiwer 
# except ImportError:
#     print("jiwer library not found. To calculate Word Error Rate, run: pip install jiwer")
#     jiwer = None

# --- Configuration ---
DATASET_DIRECTORY = "."
DATASET_NAME = "developer_typo_dataset_v4_95k.csv"
LOG_FILE = "model_evaluation_log.txt" # New log file for evaluation
SYMSPELL_DICTIONARY_PATH = os.path.join(DATASET_DIRECTORY, "developer_symspell_dict_train_only.txt") # Dict from training data

MAX_EDIT_DISTANCE_LOOKUP = 2
PREFIX_LENGTH = 7
TEST_SET_SIZE = 0.2 # 20% of data for testing
RANDOM_STATE_SPLIT = 42 # For reproducible train/test split

# --- Logger Setup ---
def setup_logger(log_file):
    logger = logging.getLogger("ModelEvaluator") # New logger name
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

# --- Text Preprocessing Functions (Using V2 from previous step) ---
def improved_tokenizer_v2(text):
    if not isinstance(text, str):
        # logger.warning(f"Expected string for tokenization, got {type(text)}. Returning empty list.") # Can be noisy
        return []
    text = text.lower()
    tokens = re.findall(r'\b[\w/:.#+-]+\b|%|[.,!?;()\'":]', text)
    return [token for token in tokens if token]

def rejoin_tokens_v2(tokens):
    if not tokens:
        return ""
    sentence = ""
    for i, token in enumerate(tokens):
        if i > 0:
            if token in ['.', ',', '!', '?', ';', ':', ')', '\'', '"'] or \
               (token == '%' and (tokens[i-1].isdigit() or tokens[i-1].endswith("d"))): # handle "50d%" if that's a pattern
                pass
            else:
                sentence += " "
        sentence += token
    sentence = re.sub(r'([(])\s+', r'\1', sentence)
    return sentence.strip()

# --- SymSpell Functions ---
from pathlib import Path

EXTRA_LEXICON_PATHS = [
    "wordlists/english_60k.txt",      # SCOWL or wordfreq words, 1 per line
    "wordlists/software-terms.dic",  # curated list of tech jargon
    # "wordlists/wordfreq-english.txt", # wordfreq words, 1 per line
]

token_re = re.compile(r"[A-Za-z0-9\+#\-\._]{2,}")   # keep +, #, ., _

def parse_software_term_line(line: str) -> list[str]:
    """
    Parses a line from software-terms.dic.
    Extracts the main term (before /) and potentially aliases.
    Returns a list of terms, all lowercased.
    """
    line = line.strip()
    if not line:
        return []
    
    terms = []
    main_term_part = line.split('/')[0].strip()
    if main_term_part:
        terms.append(main_term_part.lower())
        # Basic alias extraction (can be made more robust if needed)
        # This example just takes the main term. If you want to extract aliases
        # like "AI" from "AI/alias[AI|Artificial Intelligence]", more regex/parsing is needed.
        # For now, simplifying to just the main term for clarity.
        # If you want aliases:
        # if "/alias[" in line:
        #     try:
        #         alias_content = line.split("/alias[")[1].split("]")[0]
        #         aliases = alias_content.split('|')
        #         for alias in aliases:
        #             if alias.strip():
        #                 terms.append(alias.strip().lower())
        #     except IndexError:
        #         pass # Malformed alias string
    return list(set(terms)) # Return unique terms from this line

def create_symspell_dictionary(corpus_texts,
                               dict_path,
                               tokenizer_func): # Require tokenizer_func to be explicit
    """
    Build a SymSpell dictionary.
    - corpus_texts: iterable of strings from your main dataset's 'corrected_text'.
    - dict_path: where to write the final dictionary.txt.
    - tokenizer_func: the function to tokenize corpus_texts (e.g., improved_tokenizer_v2).
    """
    logger.info(f"Building SymSpell dictionary using tokenizer: {tokenizer_func.__name__}")
    counts = Counter()

    # 1. Process main corpus (e.g., from train_df['corrected_text'])
    # This uses the tokenizer that lowercases (improved_tokenizer_v2)
    for text in corpus_texts:
        tokens = tokenizer_func(text) # This should produce lowercased tokens
        counts.update(tokens)
    
    # 2. Process External Lexicons
    for lex_path_str in EXTRA_LEXICON_PATHS:
        lex_path = Path(lex_path_str)
        if not lex_path.exists():
            logger.warning(f"Lexicon file not found: {lex_path_str}. Skipping.")
            continue

        logger.info(f"Adding extra lexicon from {lex_path_str}")
        raw_lines = lex_path.read_text(encoding='utf-8').splitlines()
        
        for line_num, line in enumerate(raw_lines):
            line = line.strip()
            if not line or line.startswith('#'): # Skip empty lines or comments
                continue

            extracted_terms_from_line = []
            if "software-terms.dic" in lex_path_str:
                # Special parsing for software-terms.dic
                # Example: "API/alias[Application Programming Interface]" -> ["api", "application programming interface"]
                # For now, a simpler parsing: take term before '/'
                term = line.split('/')[0].strip()
                if term:
                    extracted_terms_from_line.append(term.lower())
                # Add more sophisticated alias parsing here if needed
            else: # For simple word lists like english_60k.txt
                # Assume one term per line, or term<space>frequency
                parts = line.split()
                term = parts[0]
                if term:
                   extracted_terms_from_line.append(term.lower())
            
            for t in extracted_terms_from_line:
                if t: # Ensure token is not empty
                    counts[t] += 5 # Give a slight boost to lexicon terms not in corpus, or add to existing
                                   # Adjust '5' as needed, or make it more sophisticated.
                                   # Or, if the lexicon has counts:
                                   # try: count = int(parts[1]); counts[t] += count 
                                   # except (IndexError, ValueError): counts[t] += 5 (default boost)
    
    if not counts:
        logger.error("No terms found in corpus or lexicons. Dictionary will be empty.")
        return False

    logger.info(f"Final vocabulary size (unique terms): {len(counts):,} tokens")
    
    # Write to dictionary file
    try:
        with open(dict_path, "w", encoding="utf-8") as f:
            for term, count in counts.items():
                if " " in term or not term: # Safety check: SymSpell expects single-word terms
                    logger.warning(f"Skipping invalid term found during dictionary writing: '{term}'")
                    continue
                f.write(f"{term} {int(count)}\n") # Ensure count is an integer
        logger.info(f"SymSpell dictionary written to {dict_path}")
        return True
    except Exception as e:
        logger.error(f"Error writing dictionary to {dict_path}: {e}")
        return False


def load_symspell_model(dictionary_path, max_edit_distance, prefix_length):
    logger.info("Initializing SymSpell model.")
    symspell = SymSpell(max_dictionary_edit_distance=max_edit_distance, prefix_length=prefix_length)
    if not os.path.exists(dictionary_path):
        logger.error(f"SymSpell dictionary file not found at {dictionary_path}.")
        return None
    logger.info(f"Loading dictionary from {dictionary_path} into SymSpell model.")
    if symspell.load_dictionary(dictionary_path, term_index=0, count_index=1, encoding="utf-8"):
        logger.info("SymSpell dictionary loaded successfully.")
        return symspell
    else:
        logger.error("Failed to load SymSpell dictionary.")
        return None

def correct_sentence_symspell(sentence, symspell_model, max_edit_distance_lookup, tokenizer_func, detokenizer_func):
    if not symspell_model: return sentence
    original_tokens = tokenizer_func(sentence)
    corrected_tokens = []
    keep_as_is_tokens = ['.', ',', '!', '?', ';', '(', ')', '\'', '"', ':', '%']
    for token in original_tokens:
        if token in keep_as_is_tokens:
            corrected_tokens.append(token)
            continue
        suggestions = symspell_model.lookup(token, Verbosity.CLOSEST, max_edit_distance_lookup, include_unknown=True)
        if suggestions: corrected_tokens.append(suggestions[0].term)
        else: corrected_tokens.append(token)
    return detokenizer_func(corrected_tokens)

# --- Main Script Logic for Evaluation ---
if __name__ == "__main__":
    logger.info("--- Starting Step 4: Train/Test Split and Model Evaluation ---")

    # 1. Load Full Dataset
    data_file_path = os.path.join(DATASET_DIRECTORY, DATASET_NAME)
    logger.info(f"Loading full dataset from: {data_file_path}")
    try:
        df_full = pd.read_csv(data_file_path)
        logger.info(f"Full dataset '{DATASET_NAME}' loaded successfully with {len(df_full)} rows.")
    except FileNotFoundError:
        logger.error(f"Dataset file not found at {data_file_path}.")
        exit()
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        exit()

    if 'original_text' not in df_full.columns or 'corrected_text' not in df_full.columns:
        logger.error("Dataset must contain 'original_text' and 'corrected_text' columns.")
        exit()
    
    # Ensure no NaN values in text columns that would break tokenizers
    df_full.dropna(subset=['original_text', 'corrected_text'], inplace=True)
    logger.info(f"Dataset size after dropping NaN rows (if any): {len(df_full)}")


    # 2. Split Data into Training and Test Sets
    logger.info(f"Splitting data into training and test sets (test_size={TEST_SET_SIZE})...")
    train_df, test_df = train_test_split(df_full, test_size=TEST_SET_SIZE, random_state=RANDOM_STATE_SPLIT)
    logger.info(f"Training set size: {len(train_df)} rows")
    logger.info(f"Test set size: {len(test_df)} rows")

    # 3. Create SymSpell Dictionary using ONLY the Training Set's 'corrected_text'
    if not create_symspell_dictionary(train_df['corrected_text'], SYMSPELL_DICTIONARY_PATH, improved_tokenizer_v2):
        logger.error("Failed to create SymSpell dictionary from training data. Cannot proceed with evaluation.")
        exit()

    # 4. Load SymSpell Model with the training-data-only dictionary
    symspell_model = load_symspell_model(SYMSPELL_DICTIONARY_PATH, MAX_EDIT_DISTANCE_LOOKUP, PREFIX_LENGTH)
    if not symspell_model:
        logger.error("Failed to load SymSpell model. Cannot proceed with evaluation.")
        exit()

    # 5. Evaluate on Test Set
    logger.info(f"--- Evaluating Model on Test Set ({len(test_df)} sentences) ---")
    correct_predictions = 0
    mismatched_predictions_examples = [] # Store a few examples of mismatches

    for index, row in test_df.iterrows():
        original_sentence = row['original_text']
        expected_corrected_sentence_raw = row['corrected_text']

        # Process both expected and predicted with the same tokenizer for fair comparison of content
        predicted_sentence = correct_sentence_symspell(original_sentence, symspell_model, MAX_EDIT_DISTANCE_LOOKUP, improved_tokenizer_v2, rejoin_tokens_v2)
        
        # For accuracy, we compare the tokenized versions to account for minor formatting differences
        # that our rejoiner might produce vs. the original corrected text, if any.
        # The key is whether the *words* are corrected properly.
        expected_tokens = improved_tokenizer_v2(expected_corrected_sentence_raw)
        predicted_tokens = improved_tokenizer_v2(predicted_sentence)

        if expected_tokens == predicted_tokens:
            correct_predictions += 1
        else:
            if len(mismatched_predictions_examples) < 5: # Log first 5 mismatches
                mismatched_predictions_examples.append({
                    "original": original_sentence,
                    "expected_raw": expected_corrected_sentence_raw,
                    "predicted_text": predicted_sentence,
                    "expected_tokens": expected_tokens,
                    "predicted_tokens": predicted_tokens
                })
    
    accuracy = (correct_predictions / len(test_df)) * 100 if len(test_df) > 0 else 0
    logger.info(f"\n--- Evaluation Summary ---")
    logger.info(f"Total sentences in test set: {len(test_df)}")
    logger.info(f"Number of correctly predicted sentences (token match): {correct_predictions}")
    logger.info(f"Sentence-level Accuracy: {accuracy:.2f}%")

    if mismatched_predictions_examples:
        logger.info("\n--- Examples of Mismatched Predictions (up to 5) ---")
        for i, example in enumerate(mismatched_predictions_examples):
            logger.info(f"Mismatch Example #{i+1}:")
            logger.info(f"  Original:         '{example['original']}'")
            logger.info(f"  Expected (raw):   '{example['expected_raw']}'")
            logger.info(f"  Predicted Text:   '{example['predicted_text']}'")
            # Log token differences if they are not too long for readability
            if len(str(example['expected_tokens'])) < 200 and len(str(example['predicted_tokens'])) < 200:
                 logger.debug(f"  Expected Tokens:  {example['expected_tokens']}")
                 logger.debug(f"  Predicted Tokens: {example['predicted_tokens']}")
            else:
                 logger.info("  (Token lists are too long to display here, check debug log if needed for full lists)")

    logger.info("\n--- Step 4: Model Evaluation Completed ---")
    logger.info(f"Log file generated at: {os.path.abspath(LOG_FILE)}")
    logger.info(f"SymSpell dictionary (from training data) is at: {os.path.abspath(SYMSPELL_DICTIONARY_PATH)}")