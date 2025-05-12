# backend/context_model_trainer.py
import os
import logging
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import evaluate # Hugging Face Evaluate library
import torch

# --- Configuration ---
DATASET_PATH = "./developer_typo_dataset_v4_95k.csv"  # Your main dataset
PRETRAINED_MODEL_NAME = "t5-small" # Starting point
OUTPUT_DIR = "./t5_typo_corrector_v1" # Where the fine-tuned model will be saved
LOG_FILE = os.path.join(OUTPUT_DIR, "training_log.txt")

# Training Hyperparameters (can be tuned)
TEST_SET_SIZE = 0.1  # 10% for test
VALIDATION_SET_SIZE = 0.1 # 10% of the (remaining) data for validation (e.g. 10% of 90% = 9%)
RANDOM_STATE = 42

# T5 specific prefix for the task
TASK_PREFIX = "fix typos: " 

# Model training parameters
LEARNING_RATE = 5e-5 # 2e-5 to 5e-5 is common for fine-tuning transformers
NUM_TRAIN_EPOCHS = 3 # Start with 3, can increase if needed (more epochs = longer training)
PER_DEVICE_TRAIN_BATCH_SIZE = 8 # Adjust based on your GPU memory (4, 8, 16 are common)
PER_DEVICE_EVAL_BATCH_SIZE = 8  # Adjust based on your GPU memory
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 100 # Log training loss every N steps
EVAL_STEPS = 500    # Evaluate on validation set every N steps (can be same as logging_steps)
SAVE_STEPS = 500    # Save a checkpoint every N steps
SAVE_TOTAL_LIMIT = 2 # Only keep the last N checkpoints + the best one

# --- Logger Setup ---
os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure output directory exists
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("--- Starting Context-Aware Model Fine-Tuning (T5-small) ---")

    # 1. Load and Prepare Dataset
    logger.info(f"Loading dataset from: {DATASET_PATH}")
    try:
        df_full = pd.read_csv(DATASET_PATH)
        df_full.dropna(subset=['original_text', 'corrected_text'], inplace=True)
        # For T5, we expect 'input_text' and 'target_text' columns in the Dataset object
        # or we can rename them. Let's prepare it for that.
        df_full['input_text'] = TASK_PREFIX + df_full['original_text']
        df_full['target_text'] = df_full['corrected_text']
        logger.info(f"Dataset loaded. Number of rows after dropna: {len(df_full)}")
    except FileNotFoundError:
        logger.error(f"Dataset file not found at {DATASET_PATH}. Exiting.")
        return
    except Exception as e:
        logger.error(f"Error loading dataset: {e}. Exiting.")
        return

    if len(df_full) < 100: # Arbitrary small number
        logger.error("Dataset is too small for effective training. Exiting.")
        return

    # Split data: First, separate out the test set
    train_val_df, test_df = train_test_split(
        df_full, 
        test_size=TEST_SET_SIZE, 
        random_state=RANDOM_STATE
    )
    # Then, split the remaining data into training and validation
    # Calculate validation size relative to the train_val_df
    relative_val_size = VALIDATION_SET_SIZE / (1 - TEST_SET_SIZE)
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=relative_val_size, 
        random_state=RANDOM_STATE
    )
    
    logger.info(f"Training set size: {len(train_df)}")
    logger.info(f"Validation set size: {len(val_df)}")
    logger.info(f"Test set size: {len(test_df)}")

    # Convert pandas DataFrames to Hugging Face Dataset objects
    raw_datasets = DatasetDict({
        "train": Dataset.from_pandas(train_df[['input_text', 'target_text']]),
        "validation": Dataset.from_pandas(val_df[['input_text', 'target_text']]),
        "test": Dataset.from_pandas(test_df[['input_text', 'target_text']])
    })
    logger.info(f"Raw datasets created: {raw_datasets}")

    # 2. Load Tokenizer
    logger.info(f"Loading tokenizer for '{PRETRAINED_MODEL_NAME}'")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, legacy=False) # legacy=False for T5 new tokenizers

    # 3. Preprocessing function
    def preprocess_function(examples):
        inputs = examples["input_text"]
        targets = examples["target_text"]
        
        # Tokenize inputs
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length") # Or "longest"
        
        # Tokenize targets (labels)
        # Ensure tokenizer setup for targets by using text_target argument
        labels = tokenizer(text_target=targets, max_length=128, truncation=True, padding="max_length") # Or "longest"
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    logger.info("Tokenizing datasets...")
    tokenized_datasets = raw_datasets.map(
        preprocess_function, 
        batched=True, 
        remove_columns=["input_text", "target_text"] # Remove original text columns
    )
    logger.info(f"Tokenized datasets created: {tokenized_datasets}")

    # 4. Load Pre-trained Model
    logger.info(f"Loading pre-trained model '{PRETRAINED_MODEL_NAME}'")
    model = T5ForConditionalGeneration.from_pretrained(PRETRAINED_MODEL_NAME)

    # 5. Data Collator
    # Dynamically pads sequences to the longest sequence in a batch
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    COMMON_STEP_INTERVAL = 500
    # 6. Define Training Arguments
    logger.info("Defining training arguments...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        
        eval_strategy="steps",  # <<< TRY RE-ADDING THIS
        eval_steps=COMMON_STEP_INTERVAL, # Or EVAL_STEPS if COMMON_STEP_INTERVAL isn't defined
        
        logging_strategy="steps",     # <<< TRY RE-ADDING THIS (aligns with logging_steps)
        logging_steps=LOGGING_STEPS,   
        
        save_strategy="steps",          # <<< TRY RE-ADDING THIS
        save_steps=COMMON_STEP_INTERVAL, # Or SAVE_STEPS
        
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        save_total_limit=SAVE_TOTAL_LIMIT,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        predict_with_generate=True, 
        fp16=torch.cuda.is_available(), 
        load_best_model_at_end=True, 
        metric_for_best_model="eval_loss", 
        greater_is_better=False,
        # do_eval=True # This is usually implied if evaluation_strategy="steps"
    )

    # 7. (Optional) Compute Metrics function for evaluation
    # We can add BLEU, ROUGE, CER, WER here later. For now, Trainer will report eval_loss.
    # metric = evaluate.load("sacrebleu") # Example, can add more
    # def compute_metrics(eval_preds):
    #     preds, labels = eval_preds
    #     if isinstance(preds, tuple):
    #         preds = preds[0]
    #     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    #     # Replace -100 in the labels as we can't decode them.
    #     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    #     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
    #     # Simple ROUGE and BLEU (more can be added)
    #     decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    #     decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    #     result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    #     return {k: round(v * 100, 4) for k, v in result.items()}


    # 8. Instantiate Trainer
    logger.info("Instantiating Seq2SeqTrainer...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        # compute_metrics=compute_metrics # Add this if you define compute_metrics
    )

    # 9. Start Training
    logger.info("Starting model training...")
    try:
        train_result = trainer.train()
        logger.info("Training completed.")
        
        # Save training metrics and final model
        trainer.save_model() # Saves the tokenizer too
        logger.info(f"Fine-tuned model saved to {OUTPUT_DIR}")

        metrics = train_result.metrics
        metrics["train_samples"] = len(tokenized_datasets["train"])
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logger.info(f"Training metrics saved.")

    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
        return

    # 10. (Optional) Evaluate on Test Set after training
    logger.info("Evaluating on the test set...")
    try:
        test_metrics = trainer.evaluate(eval_dataset=tokenized_datasets["test"], metric_key_prefix="test")
        test_metrics["test_samples"] = len(tokenized_datasets["test"])
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)
        logger.info("Test set evaluation metrics saved.")
    except Exception as e:
        logger.error(f"An error occurred during test set evaluation: {e}", exc_info=True)


    logger.info("--- Context-Aware Model Fine-Tuning Script Completed ---")

if __name__ == "__main__":
    # For PyTorch, check GPU availability
    
    if torch.cuda.is_available():
        logger.info(f"CUDA is available. Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("CUDA not available. Training on CPU (this will be slow).")
    
    main()