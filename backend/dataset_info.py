import pandas as pd
import great_expectations as ge
from great_expectations.expectations.expectation import (
    ExpectationConfiguration,
    Expectation,
)
from great_expectations.execution_engine import PandasExecutionEngine
from great_expectations.render.renderer.renderer import renderer
from great_expectations.render.types import RenderedStringTemplateContent
import requests # For catching specific connection errors

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import language_tool_python
import os
import time # For progress indicator
import concurrent.futures # For ThreadPoolExecutor

# --- Configuration ---
CSV_FILE_PATH = 'developer_typo_dataset_v4_95k.csv'
COSINE_SIMILARITY_THRESHOLD = 0.80
LANGUAGE_TOOL_LANG = 'en-US'
TARGET_DUMMY_ROWS = 100000 # For 100k row testing if no file exists


# --- Custom Great Expectations Classes ---

class ExpectColumnPairCosineSimilarityToBeGreaterThanOrEqualTo(Expectation):
    success_keys = ("column_A", "column_B", "threshold", "mostly")
    default_kwarg_values = {
        "column_A": None,
        "column_B": None,
        "threshold": 0.8,
        "result_format": "BASIC",
        "include_unexpected_rows": False,
        "catch_exceptions": False,
        "mostly": 1.0,
    }

    def _pandas_column_pair_cosine_similarity(
        self,
        column_A: pd.Series,
        column_B: pd.Series,
        threshold: float,
    ):
        texts_A = column_A.astype(str).fillna('')
        texts_B = column_B.astype(str).fillna('')
        all_texts_for_vocab = pd.concat([texts_A, texts_B], ignore_index=True)
        vectorizer = TfidfVectorizer()

        if all_texts_for_vocab.str.strip().eq('').all():
            return pd.Series(np.ones(len(texts_A), dtype=bool))

        try:
            vectorizer.fit(all_texts_for_vocab)
            tfidf_A = vectorizer.transform(texts_A)
            tfidf_B = vectorizer.transform(texts_B)
        except ValueError as e:
            if "empty vocabulary" in str(e).lower():
                return pd.Series(np.ones(len(texts_A), dtype=bool))
            else:
                print(f"TF-IDF Error: {e}")
                return pd.Series(np.zeros(len(texts_A), dtype=bool))

        similarities_arr = np.zeros(len(texts_A))
        for i in range(tfidf_A.shape[0]):
            if tfidf_A[i].nnz == 0 and tfidf_B[i].nnz == 0:
                similarities_arr[i] = 1.0
            elif tfidf_A[i].nnz == 0 or tfidf_B[i].nnz == 0:
                similarities_arr[i] = 0.0
            else:
                similarities_arr[i] = cosine_similarity(tfidf_A[i], tfidf_B[i])[0][0]
        
        return pd.Series(similarities_arr >= threshold)

    def _validate(
        self,
        configuration: ExpectationConfiguration,
        metrics: dict,
        runtime_configuration: dict = None,
        execution_engine: PandasExecutionEngine = None,
    ):
        column_A_name = configuration.kwargs["column_A"]
        column_B_name = configuration.kwargs["column_B"]
        threshold = configuration.kwargs["threshold"]
        mostly = configuration.kwargs.get("mostly", self.default_kwarg_values["mostly"])

        batch_data_df, _, _ = execution_engine.get_compute_domain(
            domain_kwargs=configuration.get_domain_kwargs(),
            domain_type="table"
        )
        
        if column_A_name not in batch_data_df.columns or column_B_name not in batch_data_df.columns:
            raise ValueError(f"One or both columns ({column_A_name}, {column_B_name}) not found in DataFrame.")

        series_A = batch_data_df[column_A_name]
        series_B = batch_data_df[column_B_name]

        successful_rows_series = self._pandas_column_pair_cosine_similarity(
            column_A=series_A,
            column_B=series_B,
            threshold=threshold
        )
        
        num_rows = len(batch_data_df)
        if num_rows == 0:
            return {"success": True, "result": {"observed_value": 1.0, "element_count": 0, "unexpected_count": 0}}

        unexpected_count = successful_rows_series.value_counts().get(False, 0)
        success_count = num_rows - unexpected_count
        
        success_ratio = success_count / num_rows if num_rows > 0 else 1.0
        success = success_ratio >= mostly

        return {
            "success": success,
            "result": {
                "observed_value": success_ratio,
                "element_count": num_rows,
                "unexpected_count": unexpected_count,
                "details_summary": f"{success_count}/{num_rows} passed similarity check (Threshold: ≥{threshold}, Target: ≥{mostly*100:.1f}%)"
            }
        }

class ExpectColumnValuesToHaveNoGrammarErrors(Expectation):
    expectation_type = "expect_column_values_to_have_no_grammar_errors" 
    success_keys = ("column", "mostly")
    default_kwarg_values = {
        "column": None,
        "mostly": 1.0,
        "result_format": "BASIC",
        "include_unexpected_rows": False,
        "catch_exceptions": False, # This is for GE's catching, not our internal try-except for LT
    }
    _session_lang_tool = None

    @classmethod
    def set_session_language_tool(cls, tool_instance):
        cls._session_lang_tool = tool_instance

    def _check_grammar_for_value(self, value_to_check, lang_tool_instance):
        text_to_check = ""
        if pd.notna(value_to_check) and isinstance(value_to_check, str):
            text_to_check = value_to_check
        
        if not text_to_check.strip():
            return True
        try:
            matches = lang_tool_instance.check(text_to_check)
            return len(matches) == 0
        except requests.exceptions.ConnectionError as e:
            print(f"\nWarning: LanguageTool ConnectionError for value (first 50 chars): '{str(text_to_check)[:50]}...'. Error: {e}", flush=True)
            return False
        except ConnectionResetError as e: 
            print(f"\nWarning: LanguageTool ConnectionResetError for value (first 50 chars): '{str(text_to_check)[:50]}...'. Error: {e}", flush=True)
            return False
        except (RuntimeError, AttributeError) as e: # Catch specific errors seen
            print(f"\nWarning: LanguageTool internal error ({type(e).__name__}) for value (first 50 chars): '{str(text_to_check)[:50]}...'. Error: {e}", flush=True)
            return False
        except Exception as e: # Catch-all for other unexpected LT errors
            print(f"\nWarning: Unexpected error during LanguageTool check for value (first 50 chars): '{str(text_to_check)[:50]}...'. Error: {type(e).__name__} {e}", flush=True)
            return False

    def _validate(
        self,
        configuration: ExpectationConfiguration,
        metrics: dict,
        runtime_configuration: dict = None,
        execution_engine: PandasExecutionEngine = None,
    ):
        lang_tool = ExpectColumnValuesToHaveNoGrammarErrors._session_lang_tool
        if lang_tool is None:
            raise RuntimeError(
                "LanguageTool instance has not been set. "
                "Call ExpectColumnValuesToHaveNoGrammarErrors.set_session_language_tool(tool)."
            )

        column_name = configuration.kwargs.get("column")
        mostly = configuration.kwargs.get("mostly", self.default_kwarg_values["mostly"])
        result_format = configuration.kwargs.get("result_format", self.default_kwarg_values["result_format"])
        include_unexpected_rows = configuration.kwargs.get("include_unexpected_rows", self.default_kwarg_values["include_unexpected_rows"])

        if column_name is None:
            raise ValueError("The 'column' kwarg must be specified for ExpectColumnValuesToHaveNoGrammarErrors.")

        batch_data_df, _, _ = execution_engine.get_compute_domain(
            domain_kwargs=configuration.get_domain_kwargs(),
            domain_type="table" 
        )
        
        if column_name not in batch_data_df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame.")
            
        column_series = batch_data_df[column_name]
        element_count = len(column_series)
        
        if element_count == 0:
             return {
                "success": True,
                "result": { "observed_value": 1.0, "element_count": 0, "unexpected_count": 0, },
            }

        # --- ADJUST MAX_WORKERS ---
        # Start with a low number and increase if stable.
        # If LT server becomes unstable (AttributeError: 'NoneType', ConnectionResetError), reduce this.
        num_workers = 4  # TRY STARTING WITH 2 OR 4
        
        print(f"  Starting grammar check for {element_count} rows in column '{column_name}' using up to {num_workers} worker threads...", flush=True)
        
        results_bool = [False] * element_count
        processed_count = 0
        progress_interval = max(1, element_count // 100 if element_count > 10000 else (element_count // 50 if element_count > 5000 else (element_count // 20 if element_count > 1000 else 100)))


        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_index = {
                executor.submit(self._check_grammar_for_value, column_series.iloc[i], lang_tool): i
                for i in range(element_count)
            }

            for future in concurrent.futures.as_completed(future_to_index):
                original_index = future_to_index[future]
                try:
                    is_correct = future.result()
                    results_bool[original_index] = is_correct
                except Exception as exc: 
                    print(f'\nFATAL: Task for item index {original_index} generated an exception during future.result(): {exc}', flush=True)
                    results_bool[original_index] = False 
                
                processed_count += 1
                if processed_count % progress_interval == 0 or processed_count == element_count:
                    # Using \r for a single line update. Ensure no newlines in the string.
                    print(f"    Grammar check progress: {processed_count}/{element_count} rows checked... ({processed_count/element_count*100:.1f}%) \r", end="", flush=True)
        
        print("\n  Grammar check for column '{}' completed.                            ".format(column_name), flush=True) # Newline and padding

        unexpected_count = sum(1 for is_correct in results_bool if not is_correct)
        unexpected_list = []
        if result_format != "BASIC" and include_unexpected_rows:
            for i in range(element_count):
                if not results_bool[i]:
                    unexpected_list.append(column_series.iloc[i])
        
        successful_elements = element_count - unexpected_count
        observed_value_ratio = successful_elements / element_count if element_count > 0 else 1.0
        success = observed_value_ratio >= mostly
        
        result_dict = {
            "observed_value": observed_value_ratio,
            "element_count": element_count,
            "unexpected_count": unexpected_count,
        }

        if result_format != "BASIC" and include_unexpected_rows:
            result_dict["unexpected_list"] = unexpected_list
            result_dict["partial_unexpected_list"] = unexpected_list[:20]

        return {"success": success, "result": result_dict}

    @classmethod
    @renderer(renderer_type="renderer.prescriptive")
    def _prescriptive_renderer(
        cls,
        configuration=None,
        result=None,
        language=None,
        runtime_configuration=None,
        **kwargs,
    ):
        runtime_configuration = runtime_configuration or {}
        include_column_name = runtime_configuration.get("include_column_name", True)
        styling = runtime_configuration.get("styling")
        params = configuration.kwargs
        column_name = params.get("column")
        mostly_str = ""
        if params.get("mostly") is not None and params.get("mostly") < 1.0:
            mostly_str = f", at least {params.get('mostly')*100:.1f}% of the time"
        template_str = f"must have no grammar errors{mostly_str}"
        if include_column_name and column_name:
            template_str = f"$column {template_str}"
        
        return [
            RenderedStringTemplateContent(
                content_block_type="string_template",
                string_template={ "template": template_str, "params": {"column": column_name or "N/A"}, "styling": styling, },
            )
        ]

# --- Main Script ---
def main():
    print(f"Loading dataset from: {CSV_FILE_PATH}")
    if not os.path.exists(CSV_FILE_PATH):
        print(f"Error: File not found at {CSV_FILE_PATH}")
        print(f"Creating a dummy CSV file with {TARGET_DUMMY_ROWS} rows for demonstration purposes.")
        base_dummy_data = {
            'original_text': ["Testt.", "Anotherr example.", "JavaScirpt code.", "Good.", "Good.", "", "Gramar error heree.", "This text good is.", "Check thiss one too for erors."],
            'corrected_text': ["Test.", "Another example.", "JavaScript code.", "Good.", "Good.", "", "Grammar error here.", "This text is good.", "Check this one too for errors."]
        }
        num_base_rows = len(base_dummy_data['original_text'])
        repeat_factor = (TARGET_DUMMY_ROWS + num_base_rows - 1) // num_base_rows 

        df_dummy_data = {
            'original_text': (base_dummy_data['original_text'] * repeat_factor)[:TARGET_DUMMY_ROWS],
            'corrected_text': (base_dummy_data['corrected_text'] * repeat_factor)[:TARGET_DUMMY_ROWS]
        }
        df_dummy = pd.DataFrame(df_dummy_data)
        df_dummy.to_csv(CSV_FILE_PATH, index=False)
        print(f"Dummy file '{CSV_FILE_PATH}' created with {len(df_dummy)} rows.")


    try:
        df = pd.read_csv(CSV_FILE_PATH)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    print(f"Dataset loaded with {len(df)} rows.")

    print("Initializing LanguageTool... (this might take a moment)", flush=True)
    lang_tool = None
    try:
        lang_tool_config = { 
            # 'maxTextLength': 50000, # Default is higher, adjust if needed
            # 'javaArgs': ['-Xmx512m'] # Default is -Xmx512m, increase if LT crashes due to memory with many workers
            # For many workers, you might need more RAM for the LT server, e.g., '-Xmx1g' or '-Xmx2g'
            # but be mindful of your system's total RAM.
        }
        lang_tool = language_tool_python.LanguageTool(LANGUAGE_TOOL_LANG, config=lang_tool_config)
        print("LanguageTool Initialized.", flush=True)
    except Exception as e:
        print(f"Failed to initialize LanguageTool: {e}")
        print("Grammar checks will be skipped.")


    try:
        context = ge.get_context(context_root_dir=None)
    except Exception as e:
        print(f"Error creating Great Expectations context: {e}")
        return

    datasource = context.sources.add_pandas("my_pandas_datasource")
    data_asset = datasource.add_dataframe_asset("my_dataframe_asset", dataframe=df)
    batch_request = data_asset.build_batch_request()

    validator = context.get_validator(batch_request=batch_request)

    print("\n--- Running Great Expectations Validations ---", flush=True)

    print("Test 1: Checking for duplicate rows...", flush=True)
    all_columns = df.columns.tolist()
    result_duplicates = None
    if not all_columns:
        print("  - Skipping duplicate row check as DataFrame has no columns or is empty.")
        class MockResult:
            def __init__(self, success, result_dict):
                self.success = success
                self.result = result_dict
        result_duplicates = MockResult(True, {"observed_value": "No columns to check for duplicates"})
    elif df.empty:
        print("  - Skipping duplicate row check as DataFrame is empty.")
        class MockResult:
            def __init__(self, success, result_dict):
                self.success = success
                self.result = result_dict
        result_duplicates = MockResult(True, {"observed_value": "Empty DataFrame, no duplicates"})
    else:
        result_duplicates = validator.expect_compound_columns_to_be_unique(column_list=all_columns)

    if result_duplicates:
        print(f"  - No duplicate rows: {result_duplicates.success}", flush=True)
        if not result_duplicates.success and hasattr(result_duplicates, 'result'):
            print(f"    Details: {result_duplicates.result}", flush=True)
    else:
        print("  - Duplicate row check was not performed (e.g. empty dataframe).", flush=True)


    print(f"\nTest 2: Checking cosine similarity (≥{COSINE_SIMILARITY_THRESHOLD}) ...", flush=True)
    result_cosine = None
    if df.empty:
        print("  - Skipping cosine similarity check as DataFrame is empty.")
        class MockResult:
            def __init__(self, success, result_dict):
                self.success = success
                self.result = result_dict
        result_cosine = MockResult(True, {"observed_value": "Empty DataFrame, N/A"})
    else:
        result_cosine = validator.expect_column_pair_cosine_similarity_to_be_greater_than_or_equal_to(
            column_A="original_text",
            column_B="corrected_text",
            threshold=COSINE_SIMILARITY_THRESHOLD,
            mostly=1.0
        )
    if result_cosine:
        print(f"  - Cosine similarity check: {result_cosine.success}", flush=True)
        if hasattr(result_cosine, 'result') and result_cosine.result:
            print(f"    Details: {result_cosine.result}", flush=True)

    result_grammar = None
    if lang_tool:
        ExpectColumnValuesToHaveNoGrammarErrors.set_session_language_tool(lang_tool)

        print("\nTest 3: Checking for LanguageTool grammar alerts in 'corrected_text'...", flush=True)
        if df.empty:
            print("  - Skipping grammar check as DataFrame is empty.", flush=True)
            class MockResult:
                def __init__(self, success, result_dict):
                    self.success = success
                    self.result = result_dict
            result_grammar = MockResult(True, {"observed_value": "Empty DataFrame, N/A"})
        else:
            result_grammar = validator.expect_column_values_to_have_no_grammar_errors(
                column="corrected_text",
                mostly=1.0,
                result_format="SUMMARY", 
                include_unexpected_rows=True 
            )
        if result_grammar:
            print(f"  - No grammar alerts: {result_grammar.success}", flush=True)
            if hasattr(result_grammar, 'result') and result_grammar.result:
                 print(f"    Details: {result_grammar.result}", flush=True)
                 partial_list = result_grammar.result.get("partial_unexpected_list", [])
                 if partial_list:
                     print(f"    Examples of failures (up to 5):", flush=True)
                     for item in partial_list[:5]:
                         print(f"      '{item}'", flush=True)
    else:
        print("\nTest 3: Skipped LanguageTool grammar alerts check (LanguageTool not initialized).", flush=True)

    try:
        print("\nGenerating Expectation Suite...", flush=True)
        expectation_suite = validator.get_expectation_suite(discard_failed_expectations=False)
        print("Validating full suite...", flush=True)
        validation_results_all = validator.validate(expectation_suite=expectation_suite)

        print("\n--- Overall Validation Summary ---", flush=True)
        if validation_results_all.success:
            print("All expectations PASSED!", flush=True)
        else:
            print("One or more expectations FAILED.", flush=True)
            for res_obj in validation_results_all.results:
                if not res_obj.success:
                    print(f"  - Failed Expectation: {res_obj.expectation_config.expectation_type}", flush=True)
                    print(f"    Kwargs: {res_obj.expectation_config.kwargs}", flush=True)
                    if hasattr(res_obj, 'result') and res_obj.result:
                        print(f"    Result: {res_obj.result}", flush=True)
                        partial_list_summary = res_obj.result.get("partial_unexpected_list", [])
                        if partial_list_summary:
                            print(f"    Examples of failures (up to 5):", flush=True)
                            for item in partial_list_summary[:5]:
                                 print(f"      {item}", flush=True)
    except Exception as e:
        print(f"Error during final validation or generating expectation suite: {e}", flush=True)


    if lang_tool:
        print("Closing LanguageTool...", flush=True)
        try:
            # Attempt to close, but be wary if the server process is already dead.
            # language_tool_python's close() might try to communicate with the server.
            if hasattr(lang_tool, '_process') and lang_tool._process and lang_tool._process.poll() is None:
                lang_tool.close()
                print("LanguageTool closed.", flush=True)
            else:
                print("LanguageTool server process was already terminated or not running.", flush=True)
        except Exception as e:
            print(f"Error closing LanguageTool (it might have already crashed): {e}", flush=True)

if __name__ == "__main__":
    main()