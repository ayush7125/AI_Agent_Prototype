from typing import Dict, Any, List


class Planner:
    """
    🧩 Planner Agent:
    The central orchestrator that builds a reasoning pipeline for ScholarMind.
    
    Responsibilities:
    - Analyze user intent (summary level, verification, etc.)
    - Define dynamic task sequences for other agents
    - Adapt workflow for research vs preview use cases
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.available_modes = ["research", "preview", "fact_check"]

    # -------------------------------------------------------------------------
    def plan(self, user_request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build a step-by-step action plan based on the user’s request.
        The pipeline adapts automatically depending on parameters like:
        - summary_level: 'short' or 'detailed'
        - verify: True/False
        - mode: 'research' (default), 'preview', 'fact_check'
        """

        mode = user_request.get("mode", "research").lower()
        summary_level = user_request.get("summary_level", "short").lower()
        verify = user_request.get("verify", True)

        steps = []

        # Step 1: Parse input
        steps.append({"action": "parse_pdf", "description": "Extract text and metadata from the PDF."})

        # Step 2: Chunk and embed content (index for retrieval)
        steps.append({"action": "chunk_and_index", "description": "Split content into semantically meaningful chunks and index them for retrieval."})

        # Step 3: Retrieve context for RAG (relevant background info)
        steps.append({"action": "retrieve_context", "description": "Use retriever to gather supporting evidence or related sections."})

        # Step 4: Section-wise summarization
        if summary_level == "detailed":
            steps.append({"action": "deep_section_summarize", "description": "Perform detailed section-wise summarization with subtopic breakdown."})
        else:
            steps.append({"action": "section_summarize", "description": "Perform concise summarization of major sections."})

        # Step 5: Assemble final summary
        steps.append({"action": "assemble_summary", "description": "Combine section summaries into a cohesive research summary."})

        # Step 6: Optional — Extract and verify references
        if mode == "research":
            steps.append({"action": "extract_references", "description": "Extract reference list and match citations with summary."})

        # Step 7: Optional — Review factual consistency
        if verify or mode == "fact_check":
            steps.append({"action": "review_and_verify", "description": "Review generated summary for factual consistency using retrieval-based grounding."})

        # Step 8: Format output (for UI or API response)
        steps.append({"action": "format_output", "description": "Format the final summary for UI display and API return."})

        return steps

    # -------------------------------------------------------------------------
    def describe_plan(self, steps: List[Dict[str, Any]]) -> str:
        """Return a readable, human-friendly description of the planned steps."""
        description = ["🧠 **Planned Workflow:**"]
        for i, step in enumerate(steps, start=1):
            desc = step.get("description", "No description provided.")
            description.append(f"{i}. **{step['action']}** → {desc}")
        return "\n".join(description)


# -------------------------------------------------------------------------
if __name__ == "__main__":
    planner = Planner()
    user_request = {
        "summary_level": "detailed",
        "verify": True,
        "mode": "research"
    }
    steps = planner.plan(user_request)
    print(planner.describe_plan(steps))
