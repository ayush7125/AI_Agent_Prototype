import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import clean_text, chunk_text

MODEL_PATH = "checkpoints/merged_model/checkpoint-250"  # ✅ your fine-tuned model

class Reader:
    """
    Reader Agent (ScholarMind v4.0)
    --------------------------------
    - Section-wise academic summarization agent
    - Automatically handles long text via chunking
    - Supports RAG-based evidence integration
    - Produces structured, factual summaries
    """

    def __init__(self, model_name=MODEL_PATH, device=None):
        print(f"🔹 Initializing Reader Agent with model: {model_name}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    # -------------------------------------------------------------------------
    def summarize_section(self, section_text: str, evidence: list = None, max_length: int = 250):
        """
        Summarize a given research paper section, integrating evidence when available.
        Automatically splits long sections into chunks to ensure coherent summarization.
        """
        section_text = clean_text(section_text)
        chunks = chunk_text(section_text, chunk_size=1500)  # ✅ fixed argument name

        all_summaries = []
        for i, chunk in enumerate(chunks, start=1):
            prompt = (
                f"You are a scientific summarization assistant.\n\n"
                f"Task: Write a concise, factual academic summary for the following section.\n\n"
                f"Section Chunk {i}:\n{chunk}\n"
            )

            # Include supporting evidence (if RAG retrieved)
            if evidence:
                prompt += "\nSupporting evidence:\n" + "\n".join(
                    [f"- {e['text']}" for e in evidence]
                )

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=1024
            ).to(self.device)

            with torch.no_grad():
                summary_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    num_beams=5,
                    temperature=0.7,
                    top_p=0.9,
                    length_penalty=1.0,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )

            partial_summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            all_summaries.append(clean_text(partial_summary.strip()))

        # Merge chunk summaries
        combined_summary = " ".join(all_summaries)
        return clean_text(combined_summary)

    # -------------------------------------------------------------------------
    def assemble_summary(self, sections_summaries: dict):
        """
        Combine multiple section summaries into a structured, academic-style overview.
        """
        formatted_sections = []
        for sec, txt in sections_summaries.items():
            formatted_sections.append(
                f"### {sec.capitalize()}\n"
                f"{clean_text(txt)}\n"
            )

        combined_text = "\n".join(formatted_sections)
        conclusion = (
            "\n**Overall Insight:**\n"
            "The paper cohesively connects its objectives, methods, and results, "
            "emphasizing both scientific rigor and contextual implications."
        )
        return combined_text + conclusion

    # -------------------------------------------------------------------------
    def summarize_full_document(self, full_text: str, evidence: list = None, summary_level="short"):
        """
        Summarize the entire document directly when section separation isn't clear.
        """
        print("📄 Performing full-document summarization...")
        max_len = 300 if summary_level == "short" else 600
        summary = self.summarize_section(full_text, evidence=evidence, max_length=max_len)
        return (
            f"### Full Paper Summary\n"
            f"{summary}\n\n"
            f"**Note:** This summary captures the core objectives, "
            f"methodology, results, and key takeaways of the study."
        )
