# edit_tag_spellfix/test.py
import io
import pathlib
import re

def parse_software_terms(src_path: str, dst_path: str):
    """
    Parses a list of software terms, extracts base terms and aliases,
    and writes to a SymSpell-compatible dictionary file.
    If a line in src_path has "term<space>frequency", that frequency is used.
    Otherwise, a default frequency of 1 is used.
    """
    processed_terms_with_counts = io.StringIO()
    
    src_file = pathlib.Path(src_path)
    if not src_file.exists():
        print(f"Error: Source file not found at {src_path}")
        return

    print(f"Reading terms from: {src_path}")
    for line_num, line in enumerate(src_file.read_text(encoding='utf-8').splitlines()):
        if not line.strip():
            continue
        
        # Try to parse term and optional frequency
        parts = line.strip().split()
        term_part = parts[0]
        frequency = 1 # Default frequency
        if len(parts) > 1:
            try:
                frequency = int(parts[-1]) # Assume last part might be frequency
                # Check if the part before frequency is part of a multi-word term or an alias
                # This logic assumes frequency is the VERY last part if present.
                # If aliases also contain numbers, this might need more robust parsing.
                # For "term /alias[] freq" vs "term freq /alias[]", this might need adjustment.
                # Safest if format is "BaseTerm<space>freq /alias[...]" or "BaseTerm /alias[...] freq"
                # For simplicity, let's assume frequency is the last element if a number.
                # A more robust way is to check if line.split()[-1] is purely numeric.
                
                # Check if the last part is actually a frequency or part of an alias
                # This simple check assumes frequency is space-separated and at the end
                # It might misinterpret "GPT-3 50" if "GPT-3" is the term and "50" is freq.
                # A more robust regex for term extraction before frequency might be needed if format is complex.

                # Let's refine: if the last part is a number AND it's not part of the alias block.
                potential_freq_candidate = line.split()[-1]
                if potential_freq_candidate.isdigit():
                    # Check if it's outside an alias block
                    if f" {potential_freq_candidate}" in line and not re.search(r'\[.*' + re.escape(potential_freq_candidate) + r'.*\]', line):
                         # It's likely a frequency
                        frequency = int(potential_freq_candidate)
                        term_part_to_process = " ".join(line.split()[0:-1]) # Everything except the last part
                    else: # Number is inside alias or it's just the term itself
                        term_part_to_process = line.strip() # Process the whole line as term block
                        frequency = 1 # Reset to default if number was part of term/alias
                else:
                    term_part_to_process = line.strip() # No numeric last part
                    frequency = 1

            except ValueError:
                term_part_to_process = line.strip() # Last part wasn't a valid integer
                frequency = 1
        else:
            term_part_to_process = line.strip() # Single part, must be the term
            frequency = 1
            
        base, *rest = re.split(r'/[a-z]+\[', term_part_to_process, maxsplit=1, flags=re.I)
        terms_on_line = [base.strip()]
        
        if rest:
            alias_block_content = rest[0].rstrip(']')
            terms_on_line.extend([t.strip() for t in alias_block_content.split('|')])
        
        for term_entry in terms_on_line:
            if term_entry:
                # Original script had: cleaned_term = term_entry.replace(' ', '')
                # For SymSpell, it's often better to keep spaces for multi-word terms if your lookup will use them,
                # or remove them if you want "fullstack" to be a single token.
                # Let's keep spaces for now, as SymSpell can handle them if dictionary is built that way.
                # However, the original 'cleaned_term' logic removing spaces was there.
                # If you prefer space-removed terms, revert to:
                # cleaned_term = term_entry.replace(' ', '')
                cleaned_term = term_entry # Keep spaces
                if cleaned_term:
                    processed_terms_with_counts.write(f"{cleaned_term} {frequency}\n")

    output_content = processed_terms_with_counts.getvalue()
    if output_content:
        print(f"Writing processed dictionary to: {dst_path}")
        with open(dst_path, 'w', encoding='utf-8') as f_out:
            f_out.write(output_content)
        print(f"Successfully created {dst_path} with processed terms.")
    else:
        print(f"Warning: No terms were processed from {src_path}. Output file {dst_path} will be empty or not created.")
    processed_terms_with_counts.close()

if __name__ == "__main__":
    source_dictionary = "wordlists/software-terms.txt"
    output_dictionary = "wordlists/software-terms-clean.dic"
    print(f"Attempting to parse '{source_dictionary}' into '{output_dictionary}'...")
    parse_software_terms(source_dictionary, output_dictionary)