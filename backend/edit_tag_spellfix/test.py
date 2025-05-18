
from collections import Counter
import io
import pathlib
import re


def parse_software_terms(src_path: str, dst_path: str):
    out = io.StringIO()
    for line in pathlib.Path(src_path).read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        # Split on '/alias[' or '/js[' etc.x
        base, *rest = re.split(r'/[a-z]+\[', line, maxsplit=1, flags=re.I)
        terms = [base.strip()]
        if rest:
            alias_block = rest[0].rstrip(']')
            terms += [t.strip() for t in alias_block.split('|')]
        for term in terms:
            # Strip spaces in multi-word aliases for SymSpell token (optional)
            cleaned = term.replace(' ', '')
            out.write(f"{cleaned} 1\n")
    counts = Counter()
    with open(path_in, encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            term, freq = line.split()
            counts[term] += int(freq)
    with open(path_out, 'w', encoding='utf-8') as g:
        for term, freq in counts.items():
            g.write(f"{term} {freq}\n")


parse_software_terms("wordlists/software-terms.txt", "wordlists/software-terms-clean.dic")
