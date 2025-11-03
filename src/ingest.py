import fitz  # PyMuPDF
from utils import clean_text, chunk_text


def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    return "\n".join(pages)


def split_sections_by_headings(text: str) -> dict:
    # heuristic: split by newline headings that are uppercase or known section names
    sections = {}
    current = 'front'
    buffer = []
    for line in text.split('\n'):
        if len(line.strip())>0 and (line.strip().isupper() or any(k in line.lower() for k in ['introduction','method','results','conclusion','references','abstract'])):
            if buffer:
                sections[current] = '\n'.join(buffer)
                buffer = []
            current = line.strip()
        else:
            buffer.append(line)
    if buffer:
        sections[current] = '\n'.join(buffer)
    return sections


if __name__ == '__main__':
    import sys
    p = sys.argv[1]
    text = extract_text_from_pdf(p)
    secs = split_sections_by_headings(text)
    print(list(secs.keys()))
