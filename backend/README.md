# Backend Instructions

## 1) Setting up venv

```
python -m venv venv

# for mac
source venv/bin/activate

    
# for windows
.venv/Scripts/activate

```

## 2) Install dependencies

```
pip install -r requirements.txt
```

## Context-Aware Model (T5 Fine-Tuning) Workflow

This section details the steps after successfully fine-tuning a context-aware model (e.g., T5) using `context_model_trainer.py`. The fine-tuned model will be saved in the directory specified by `OUTPUT_DIR` in that script (e.g., `./t5_typo_corrector_v1`).

### Step 1: Qualitative Evaluation & Inference Testing

Before fully integrating, it's crucial to test the fine-tuned T5 model qualitatively to see how it performs on various inputs, especially your target sentences and edge cases.

1.  **Locate the Inference Script/Module:**
    We have created `t5_corrector_api_module.py` for this purpose. This script can load the fine-tuned T5 model and perform predictions.

2.  **Configure Model Path in `t5_corrector_api_module.py`:**
    Ensure the `MODEL_PATH` variable in `t5_corrector_api_module.py` points to the correct output directory where your fine-tuned T5 model was saved by `context_model_trainer.py`. For example:
    ```python
    # In t5_corrector_api_module.py
    MODEL_PATH = "./t5_typo_corrector_v1" # Or your specific output directory
    MODEL_NAME_FOR_API = "T5-small_DevTypo_FineTuned_V1" # Update if needed
    ```

3.  **Run Standalone Test:**
    Execute `t5_corrector_api_module.py` directly to test its inference capabilities with the sample sentences provided in its `if __name__ == '__main__':` block.
    ```bash
    # Ensure your Python virtual environment is activated
    # source venv/bin/activate 
    python t5_corrector_api_module.py
    ```
    Modify the `test_sentences` list in the script to include:
    *   Your primary example sentence (e.g., "a machine learning and fuil stack...").
    *   Sentences that SymSpell struggled with.
    *   Correctly spelled sentences (to see if the model over-corrects).
    *   Sentences with various types of typos.

4.  **Analyze Output:**
    *   Does the T5 model correct your target sentence as expected?
    *   How does it handle different types of typos?
    *   Is it making minimal edits, or is it too generative (rewriting sentences more than desired)?
    *   Note down any systematic errors or areas where its corrections are not ideal.

5.  **Experiment with Generation Parameters (in `t5_corrector_api_module.py`):**
    Inside the `correct_text_t5` function in `t5_corrector_api_module.py`, you can adjust the `t5_model.generate()` parameters:
    *   `num_beams`: Try `1` (greedy decoding) vs. `4` or `5` (beam search). Greedy might be more literal.
    *   `max_length`: Ensure it's sufficient for your expected corrected outputs.
    *   Other parameters like `length_penalty`, `no_repeat_ngram_size` can be explored if you see issues with output length or repetition.
    Iterate on these parameters and re-run the standalone test to see their effect.

### Step 2: Quantitative Evaluation (Advanced Metrics)

While the training script provides loss, more specific metrics are useful for typo correction.

1.  **Modify `context_model_trainer.py` for Richer Evaluation:**
    *   Uncomment and complete the `compute_metrics` function within `context_model_trainer.py`. You'll need libraries like `evaluate`, `jiwer` (for CER/WER), `rouge_score`, and `sacrebleu`.
    *   Ensure these libraries are in your `requirements.txt` and installed (`pip install evaluate jiwer sacrebleu rouge_score nltk`). You'll also need `nltk` for sentence tokenization for some metrics. Add `import nltk` and `import numpy as np` to `context_model_trainer.py`.
    ```python
    # Example compute_metrics function (add to context_model_trainer.py)
    import numpy as np
    import nltk # Add this import
    # Ensure you have: from evaluate import load as evaluate_load (or just evaluate.load)

    # Load metrics - do this globally or once at the start of main()
    # bleu_metric = evaluate.load("sacrebleu")
    # rouge_metric = evaluate.load("rouge")
    # wer_metric = evaluate.load("wer")
    # cer_metric = evaluate.load("cer")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds
        
        # Decode generated summaries, replacing -100 in the labels as tokenizer.pad_token_id.
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Prepare for ROUGE and BLEU (nltk sentence tokenization)
        decoded_preds_rouge_bleu = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels_rouge_bleu = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = {}
        
        # BLEU score
        bleu_score = evaluate.load("sacrebleu").compute(predictions=decoded_preds_rouge_bleu, references=[[l] for l in decoded_labels_rouge_bleu])
        result["bleu"] = bleu_score["score"]

        # ROUGE score
        rouge_scores = evaluate.load("rouge").compute(predictions=decoded_preds_rouge_bleu, references=decoded_labels_rouge_bleu)
        result.update(rouge_scores) # Adds rouge1, rouge2, rougeL, rougeLsum
        
        # WER
        wer_score = evaluate.load("wer").compute(predictions=decoded_preds, references=decoded_labels)
        result["wer"] = wer_score
        
        # CER
        cer_score = evaluate.load("cer").compute(predictions=decoded_preds, references=decoded_labels)
        result["cer"] = cer_score

        # Prediction lengths
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()}
    ```
    *   Pass this function to the `Seq2SeqTrainer`:
        ```python
        # In context_model_trainer.py, when instantiating Trainer:
        trainer = Seq2SeqTrainer(
            # ... other args ...
            compute_metrics=compute_metrics 
        )
        ```

2.  **Re-run Evaluation on Test Set (if needed):**
    If you didn't have `compute_metrics` during the main training, you can load your saved model and run evaluation:
    ```python
    # At the end of main() in context_model_trainer.py, or in a separate script:
    # logger.info("Loading best model for final test set evaluation with metrics...")
    # model = T5ForConditionalGeneration.from_pretrained(OUTPUT_DIR) # Loads best model if saved
    # tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, legacy=False)
    #
    # trainer_for_eval = Seq2SeqTrainer( # Re-instantiate trainer with the loaded model
    #     model=model,
    #     args=training_args, # Use the same training_args
    #     eval_dataset=tokenized_datasets["test"],
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    #     compute_metrics=compute_metrics
    # )
    # test_metrics = trainer_for_eval.evaluate(eval_dataset=tokenized_datasets["test"], metric_key_prefix="final_test")
    # trainer_for_eval.log_metrics("final_test", test_metrics)
    # trainer_for_eval.save_metrics("final_test", test_metrics)
    ```
    (The main training script already has an evaluation step for the test set if `compute_metrics` was defined before `trainer.train()` and `trainer.evaluate(eval_dataset=tokenized_datasets["test"]...)` is called.)

3.  **Analyze Metrics:**
    *   **BLEU/ROUGE:** Higher is better (closer to 1.0 or 100). Indicates overlap with reference.
    *   **WER/CER:** Lower is better (closer to 0). Indicates fewer word/character errors.
    *   These give a more nuanced view than just loss or sentence accuracy.

### Step 3: Integrate the Fine-Tuned T5 Model into the Flask API

This involves modifying `app.py` to use `t5_corrector_api_module.py`.

1.  **Ensure `t5_corrector_api_module.py` is Ready:**
    *   The `MODEL_PATH` should point to your successfully fine-tuned T5 model (e.g., `./t5_typo_corrector_v1`).
    *   The `load_t5_model_and_tokenizer()` function should work correctly.
    *   The `correct_text_t5()` function should take a sentence and return the corrected version, processing time, and whether corrections were made.

2.  **Modify `app.py`:**
    *   **Comment out/Remove SymSpell:**
        *   Remove `import model_loader` (for SymSpell).
        *   Remove the SymSpell model loading block `with app.app_context(): ... model_loader.load_model() ...`.
    *   **Import T5 Corrector:**
        ```python
        import t5_corrector_api_module
        ```
    *   **Load T5 Model on Startup:**
        ```python
        # In app.py
        with app.app_context():
            if not t5_corrector_api_module.load_t5_model_and_tokenizer():
                app.logger.error("CRITICAL: Fine-tuned T5 model failed to load on application startup.")
            else:
                app.logger.info("Fine-tuned T5 model loaded successfully on application startup.")
        ```
    *   **Update Health Check:**
        Modify `/api/health` to check `t5_corrector_api_module.t5_model` and `t5_corrector_api_module.t5_tokenizer`.
        ```python
        # In app.py, inside health_check()
        if t5_corrector_api_module.t5_model is not None and t5_corrector_api_module.t5_tokenizer is not None:
            return jsonify({"status": "healthy", "message": "Backend is running and T5 model is loaded."}), 200
        else:
            # ... error ...
        ```
    *   **Update `/api/check_typos` Endpoint:**
        *   Check if the T5 model is loaded.
        *   Call `t5_corrector_api_module.correct_text_t5(original_sentence)`.
        *   Use `t5_corrector_api_module.MODEL_NAME_FOR_API` in the response.
        *   Update the log messages and response message to indicate T5 is being used.
        ```python
        # In app.py, inside check_typos_endpoint()
        if t5_corrector_api_module.t5_model is None or t5_corrector_api_module.t5_tokenizer is None:
            app.logger.error("T5 Model is not loaded. Cannot process request.")
            # ... return error ...

        # ... (get original_sentence) ...

        app.logger.info(f"Processing sentence with T5 model: '{original_sentence}'")
        
        corrected_sentence, processing_time_ms, corrections_made = \
            t5_corrector_api_module.correct_text_t5(original_sentence)
        
        app.logger.info(f"Original: '{original_sentence}', T5 Corrected: '{corrected_sentence}', Time: {processing_time_ms:.2f}ms, Corrections Made: {corrections_made}")

        response_data = {
            "original_sentence": original_sentence,
            "corrected_sentence": corrected_sentence,
            "model_name": t5_corrector_api_module.MODEL_NAME_FOR_API,
            # ... rest of the response ...
            "message": "Typos checked successfully with T5 model."
        }
        return jsonify(response_data), 200
        ```

3.  **Restart the Flask API Server:**
    ```bash
    python app.py
    ```

4.  **Test Thoroughly with Frontend:**
    *   Verify the model name displayed in the frontend.
    *   Test various sentences.
    *   Monitor backend logs for any errors from the T5 model.
    *   Be mindful of inference time, especially if running the T5 model on a CPU (it will be noticeably slower than SymSpell).

### Step 4: Iteration and Further Improvements

Based on the qualitative and quantitative results:

*   **Dataset Refinement:** If T5 is still making undesired changes or failing on key typos:
    *   **Focus on `corrected_text` quality:** Ensure it represents the *exact* minimal correction.
    *   **Augment with specific examples:** If T5 fails on "fuil" -> "full-stack", add more training examples where `original_text` has "fuil" (and variations like "ful", "fll-stck") and `corrected_text` is precisely "full-stack engineer" (or just "full-stack" if you want it to correct only that word).
    *   **Negative Examples (Advanced):** For some sequence-to-sequence tasks, people add examples where the input is *already correct* and the target is the same. This can sometimes teach the model to be less aggressive in changing already correct text, but it needs to be balanced.
*   **Hyperparameter Tuning for T5:** If results are suboptimal, learning rate, batch size, number of epochs, or even the choice of pre-trained model (e.g., trying `t5-base` if `t5-small` isn't capturing enough, though this increases training time) can be adjusted.
*   **More Sophisticated Generation Strategies:** Explore different decoding strategies in `t5_model.generate()` beyond just `num_beams`.

This iterative process of fine-tuning, evaluating, and refining the data/model is standard for achieving high performance with complex models like T5.