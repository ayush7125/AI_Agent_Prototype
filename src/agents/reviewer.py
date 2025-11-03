"""
src/agents/reviewer.py — Reviewer Agent (self-reflection, factual grounding, and quality scoring)
"""

from typing import List, Dict, Any
from difflib import SequenceMatcher
import numpy as np


class Reviewer:
    """
    🧠 Reviewer Agent:
    Evaluates the factual reliability and grounding of model-generated summaries.
    Cross-checks claims with retrieved context from the RAG retriever.
    """

    def __init__(self, retriever, model=None):
        """
        Args:
            retriever: A retrieval engine (e.g., Chroma retriever) for evidence lookup
            model: Optional LM (for re-ranking or natural language scoring)
        """
        self.retriever = retriever
        self.model = model

    # -------------------------------------------------------------------------
    def ground_claim(self, claim: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Returns supporting evidence chunks and a confidence score.
        Uses both lexical and semantic overlap with retrieved passages.
        """
        hits = self.retriever.query(claim, k=top_k)
        supporting = []

        for h in hits:
            text = h.get("text", "")
            lexical_overlap = self._lexical_overlap(claim, text)
            semantic_score = self._semantic_similarity(claim, text)

            # Confidence fusion (weighted average)
            confidence = round(0.6 * lexical_overlap + 0.4 * semantic_score, 3)

            if confidence > 0.35:  # threshold for support
                supporting.append({
                    "source": h.get("source", "unknown"),
                    "text": text,
                    "confidence": confidence
                })

        overall_confidence = round(
            np.mean([s["confidence"] for s in supporting]) if supporting else 0.0, 3
        )

        return {
            "claim": claim,
            "supporting": supporting,
            "overall_confidence": overall_confidence,
            "verdict": self._get_verdict(overall_confidence)
        }

    # -------------------------------------------------------------------------
    def review_summary(self, summary: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Split the generated summary into sentences, ground each claim,
        and produce an aggregated factuality report.
        """
        claims = [s.strip() for s in summary.split(".") if len(s.strip()) > 10]
        results = []

        for claim in claims:
            grounded = self.ground_claim(claim, top_k=top_k)
            results.append(grounded)

        avg_conf = np.mean([r["overall_confidence"] for r in results]) if results else 0
        factuality_score = round(avg_conf * 100, 2)

        return {
            "summary_evaluation": {
                "total_claims": len(results),
                "avg_confidence": avg_conf,
                "factuality_score": factuality_score,
                "verdict": self._get_verdict(avg_conf),
            },
            "claim_level_evidence": results
        }

    # -------------------------------------------------------------------------
    def _lexical_overlap(self, a: str, b: str) -> float:
        """Computes normalized token overlap between two texts."""
        a_tokens = set(a.lower().split())
        b_tokens = set(b.lower().split())
        if not a_tokens or not b_tokens:
            return 0.0
        return len(a_tokens.intersection(b_tokens)) / len(a_tokens)

    def _semantic_similarity(self, a: str, b: str) -> float:
        """
        Simple semantic similarity proxy using sequence matcher.
        You can replace this with sentence-transformer embeddings later.
        """
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def _get_verdict(self, confidence: float) -> str:
        """Converts numeric confidence into qualitative verdict."""
        if confidence >= 0.75:
            return "✅ Well-grounded"
        elif confidence >= 0.5:
            return "⚠️ Partially supported"
        else:
            return "❌ Weak or unsupported"

    # -------------------------------------------------------------------------
    def generate_report(self, review_output: Dict[str, Any]) -> str:
        """
        Converts structured factuality output into human-readable Markdown report.
        Useful for frontend display.
        """
        summary_eval = review_output.get("summary_evaluation", {})
        claims = review_output.get("claim_level_evidence", [])

        report = [
            "## 🧾 Factuality Review Report",
            f"- **Total Claims:** {summary_eval.get('total_claims', 0)}",
            f"- **Average Confidence:** {summary_eval.get('avg_confidence', 0):.2f}",
            f"- **Factuality Score:** {summary_eval.get('factuality_score', 0)}%",
            f"- **Overall Verdict:** {summary_eval.get('verdict', 'N/A')}",
            "\n### 🔍 Claim-level Analysis:\n"
        ]

        for i, c in enumerate(claims, 1):
            report.append(
                f"{i}. **Claim:** {c['claim']}\n"
                f"   - Verdict: {c['verdict']}\n"
                f"   - Confidence: {c['overall_confidence']}\n"
                f"   - Supporting Evidence: {len(c['supporting'])} snippets"
            )
        return "\n".join(report)


# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("🧠 Reviewer demo – requires retriever instance")
