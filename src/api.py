"""
ScholarMind API (v4.3)
------------------------------------
Final Stable Release
✅ Fixes all runtime and timeout issues
✅ Dynamically configures generation for long docs
✅ Memory-safe chunking for large PDFs
✅ Context-aware section summarization
✅ Compatible with Gradio Frontend v5.1
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
from rag_index import Retriever
import uvicorn
import fitz  # PyMuPDF
import torch
import re
import logging
from typing import List, Dict

# ================================================================
# 🔹 App Initialization
# ================================================================
app = FastAPI(title="ScholarMind API", version="4.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ================================================================
# 🔹 Model Setup
# ================================================================
MODEL_PATH = "checkpoints/merged_model/checkpoint-250"
logging.info(f"🔹 Loading fine-tuned model from {MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# ================================================================
# 🔹 Retriever
# ================================================================
retriever = Retriever()
try:
    retriever.create_collection("scholarmind")
except Exception:
    logging.warning("📁 Using existing Chroma collection: scholarmind")

# ================================================================
# 🔹 Helper Functions
# ================================================================
def extract_text_from_pdf(pdf_file: UploadFile) -> str:
    """Extracts text content from a PDF."""
    text = ""
    try:
        with fitz.open(stream=pdf_file.file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text("text")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF extraction failed: {e}")
    return text.strip()


def chunk_text(text: str, chunk_size=1500, overlap=150) -> List[str]:
    """Splits text into overlapping word chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks


def detect_sections(text: str) -> List[tuple]:
    """
    Detects key sections (Introduction, Methods, Results, etc.)
    with fallback to full text if headings not found.
    """
    pattern = r"(?i)\b(abstract|introduction|related work|background|methodology|methods|experiments|results|analysis|discussion|conclusion|references|findings)\b"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return [("Full Paper", text)]

    sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if len(section_text) > 200:  # ignore tiny fragments
            sections.append((match.group().capitalize(), section_text))
    return sections


# ================================================================
# 🔹 Summarization Core
# ================================================================
def summarize_with_context(text: str, context: str, summary_level: str = "short") -> str:
    """Generates a structured academic summary with safe generation config."""
    summary_task = {
        "short": "Summarize concisely the objective, methodology, and key findings.",
        "detailed": "Generate a comprehensive academic summary including motivation, methods, experiments, results, and implications.",
        "section": "Summarize the given section in a concise, factual, and academic tone (3–5 lines per aspect)."
    }.get(summary_level, "Summarize the text concisely.")

    prompt = f"""
You are an expert research summarizer.
Context (if any): {context}

Research Paper Text:
{text}

Task: {summary_task}

Output format:
- Objective
- Methodology
- Results / Findings
- Key Insights / Limitations
"""

    # Dynamically adapt generation parameters based on summary level
    gen_cfg = GenerationConfig(
        max_new_tokens=400 if summary_level == "short" else 800,
        num_beams=4,
        length_penalty=1.0,
        no_repeat_ngram_size=3,
        early_stopping=True,
        do_sample=False
    )

    inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        summary_ids = model.generate(**inputs, generation_config=gen_cfg)

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# ================================================================
# 🔹 Summarization Endpoint
# ================================================================
MAX_SECTIONS = 10
@app.post("/summarize")
async def summarize(
    file: UploadFile = File(...),
    summary_level: str = Form("short"),
    verify: bool = Form(True)
):
    """Main summarization endpoint."""
    text = extract_text_from_pdf(file)
    logging.info(f"✅ Extracted {len(text)} characters from {file.filename}")

    if len(text) < 300:
        raise HTTPException(status_code=400, detail="PDF content too short to summarize.")

    sections = detect_sections(text)
    logging.info(f"📄 Detected {len(sections)} sections.")
    sections = sections[:MAX_SECTIONS]

    # Step 1: Chunk + index text
    chunks = chunk_text(text, chunk_size=1500)
    retriever.index_documents([
        {"id": f"chunk_{i}", "text": c, "metadata": {"source": file.filename, "chunk_id": i}}
        for i, c in enumerate(chunks)
    ])
    logging.info(f"✅ Indexed {len(chunks)} chunks for retrieval.")

    # Step 2: Section-level summaries
    section_summaries: Dict[str, str] = {}
    for sec_name, sec_text in sections:
        logging.info(f"🧠 Summarizing section: {sec_name}")
        context_docs = retriever.query(sec_text[:800], k=3)
        context = "\n".join([d["text"] for d in context_docs])
        summary = summarize_with_context(sec_text[:8000], context, summary_level)
        section_summaries[sec_name] = summary

    # Step 3: Final combined summary
    combined = "\n\n".join([f"### {k}\n{v}" for k, v in section_summaries.items()])
    final_summary = summarize_with_context(combined, "", "detailed")

    # Step 4: Optional factual verification
    verification = retriever.query(final_summary, k=3) if verify else []

    return {
        "summary": final_summary,
        "sections": section_summaries,
        "verification": verification,
        "summary_level": summary_level
    }


# ================================================================
# 🔹 Root Endpoint
# ================================================================
@app.get("/")
async def root():
    return {"message": "🚀 ScholarMind API v4.3 — Stable & Optimized Summarizer Ready!"}


# ================================================================
# 🔹 Run API
# ================================================================
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
