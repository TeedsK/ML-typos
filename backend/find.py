import importlib.resources
try:
    dictionary_path = importlib.resources.files("symspellpy").joinpath("frequency_dictionary_en_82_765.txt")
    if dictionary_path.exists():
        print(f"Found dictionary at: {dictionary_path}")
    else:
        print("Bundled dictionary not found at the expected location.")
except Exception as e:
    print(f"Error trying to locate bundled dictionary: {e}")