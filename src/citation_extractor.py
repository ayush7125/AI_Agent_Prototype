import requests
from typing import List, Dict
import re

GROBID_URL = 'http://localhost:8070'


def parse_references_with_grobid(pdf_path: str) -> List[Dict]:
    files = {'input': open(pdf_path, 'rb')}
    r = requests.post(GROBID_URL + '/api/processReferences', files=files)
    # grobid returns TEI XML. minimal parse example; in production use grobid-client
    return r.text


def naive_reference_split(text: str) -> List[str]:
    # fallback: split by numeric bullets or newline patterns
    parts = re.split(r'\n\s*\[?\d+\]?\s+', text)
    return [p.strip() for p in parts if len(p.strip())>20]

if __name__ == '__main__':
    print('GROBID sample fetch not run in demo')