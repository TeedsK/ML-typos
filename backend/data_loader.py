import pandas as pd
import logging
import os
from datetime import datetime

# --- Configuration ---
# <<< IMPORTANT: Please update this path if your dataset is in a different directory >>>
DATASET_DIRECTORY = "."  # Assuming the script is in the same directory as the dataset files
DATASET_NAME = "developer_typo_dataset_1000.csv" # Starting with the smallest dataset
LOG_FILE = "data_loading_log.txt"

# --- Logger Setup ---
def setup_logger(log_file):
    """Sets up a logger that writes to both console and a file."""
    logger = logging.getLogger("DatasetLoader")
    logger.setLevel(logging.DEBUG)  # Capture all levels of logs

    # Avoid adding multiple handlers if logger already has them
    if not logger.handlers:
        # File Handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        formatter_fh = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter_fh)
        logger.addHandler(fh)

        # Console Handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO) # Only INFO and above for console
        formatter_ch = logging.Formatter('%(levelname)s: %(message)s')
        ch.setFormatter(formatter_ch)
        logger.addHandler(ch)
    return logger

logger = setup_logger(LOG_FILE)

# --- Main Script ---
def load_and_inspect_dataset(directory, filename):
    """Loads the dataset and performs initial inspection."""
    file_path = os.path.join(directory, filename)
    logger.info(f"Attempting to load dataset from: {file_path}")

    try:
        # Attempt to load with UTF-8 encoding first, which is common.
        df = pd.read_csv(file_path, encoding='utf-8')
    except FileNotFoundError:
        logger.error(f"Dataset file not found at {file_path}. Please check the path and filename.")
        return None
    except UnicodeDecodeError:
        logger.warning(f"UTF-8 decoding failed for {file_path}. Attempting with 'latin1' encoding.")
        try:
            df = pd.read_csv(file_path, encoding='latin1')
        except Exception as e:
            logger.error(f"Failed to load dataset with 'latin1' encoding as well. Error: {e}")
            return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading the dataset: {e}")
        return None

    logger.info(f"Dataset '{filename}' loaded successfully.")
    logger.info("--- First 5 Rows (Head) ---")
    logger.info("\n" + df.head().to_string())

    logger.info("\n--- Dataset Information (Info) ---")
    # Redirect print output of df.info() to logger
    # This is a bit of a workaround as df.info() prints to stdout directly
    import io
    buffer = io.StringIO()
    df.info(buf=buffer)
    logger.info("\n" + buffer.getvalue())


    logger.info("\n--- Basic Descriptive Statistics ---")
    logger.info("\n" + df.describe(include='all').to_string())

    logger.info("\n--- Missing Value Check ---")
    missing_values = df.isnull().sum()
    logger.info("\n" + missing_values.to_string())

    if missing_values.sum() == 0:
        logger.info("No missing values found in the dataset. Great!")
    else:
        logger.warning(f"Found {missing_values.sum()} total missing values. Further investigation needed.")
        for col, count in missing_values.items():
            if count > 0:
                logger.warning(f"Column '{col}' has {count} missing values.")

    # Validate column names
    expected_columns = ['original_text', 'corrected_text']
    if list(df.columns) == expected_columns:
        logger.info(f"Column names {expected_columns} are as expected.")
    else:
        logger.error(f"Column names are NOT as expected. Found: {list(df.columns)}, Expected: {expected_columns}")
        return None # Critical error if columns are not what we expect

    logger.info(f"Number of rows: {len(df)}")
    logger.info(f"Number of columns: {len(df.columns)}")

    return df

if __name__ == "__main__":
    logger.info("--- Starting Step 2: Data Loading and Initial Inspection ---")
    dataset_df = load_and_inspect_dataset(DATASET_DIRECTORY, DATASET_NAME)

    if dataset_df is not None:
        logger.info("--- Data Loading and Initial Inspection Step Completed Successfully ---")
        # You can now work with dataset_df
        # For example, print a random sample:
        # logger.info("\n--- Random Sample (5 rows) ---")
        # logger.info("\n" + dataset_df.sample(min(5, len(dataset_df))).to_string())
    else:
        logger.error("--- Data Loading and Initial Inspection Step Failed ---")

    logger.info(f"Log file generated at: {os.path.abspath(LOG_FILE)}")