"""
ScholarMind Gradio Frontend (v5.1)
----------------------------------
Elegant academic UI for ScholarMind Research Paper Summarizer.
Features:
- Upload & summarize research PDFs
- Select summary detail level
- Optional factual verification
- Robust long-timeout support for large papers
- Academic-style clean layout
"""

import gradio as gr
import requests
import os
import threading

# ================================================================
# 🔹 API Configuration
# ================================================================
API_URL = "http://127.0.0.1:8000/summarize"  # FastAPI backend


# ================================================================
# 🔹 Backend Request Handler
# ================================================================
def summarize_pdf(file, summary_level, verify):
    """Handles user file upload and interacts with FastAPI backend."""
    if file is None:
        return "⚠️ Please upload a PDF file to summarize."

    result_container = {"result": "⏳ Processing PDF... please wait while the summary is generated."}

    def process_request():
        try:
            with open(file.name, "rb") as f:
                files = {"file": (os.path.basename(file.name), f, "application/pdf")}
                data = {"summary_level": summary_level, "verify": str(verify).lower()}

                # Allow up to 1 hour for long papers
                response = requests.post(API_URL, files=files, data=data, timeout=3600)

            if response.status_code == 200:
                try:
                    resp = response.json()
                    if "error" in resp:
                        result_container["result"] = f"❌ Error: {resp['error']}"
                        return

                    summary = resp.get("summary", "").strip()
                    sections = resp.get("sections", {})
                    verification = resp.get("verification", [])

                    formatted_output = "## 📘 Research Paper Summary\n\n"

                    if sections:
                        for sec, text in sections.items():
                            formatted_output += f"### {sec}\n{text.strip()}\n\n"

                    formatted_output += f"---\n\n### 🧾 Consolidated Summary\n{summary}\n\n"

                    if verification:
                        formatted_output += "### 🔍 Top Supporting Evidence\n"
                        for v in verification[:3]:
                            snippet = v.get('text', '')[:300].strip()
                            formatted_output += f"- {snippet}...\n"
                        formatted_output += "\n"

                    result_container["result"] = formatted_output.strip()

                except Exception as parse_err:
                    result_container["result"] = f"⚠️ Failed to parse API response: {parse_err}"

            else:
                result_container["result"] = f"❌ API Error {response.status_code}: {response.text}"

        except requests.exceptions.ReadTimeout:
            result_container["result"] = "⚠️ Request timed out. The summary generation took too long. Try selecting a shorter summary type."
        except requests.exceptions.ConnectionError:
            result_container["result"] = "🚫 Unable to connect to the ScholarMind API. Ensure the FastAPI backend is running."
        except Exception as e:
            result_container["result"] = f"⚠️ Unexpected error: {str(e)}"

    # Background processing
    thread = threading.Thread(target=process_request)
    thread.start()
    thread.join()

    return result_container["result"]


# ================================================================
# 🔹 Custom Theme
# ================================================================
theme = gr.themes.Soft(
    primary_hue="orange",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
).set(
    button_primary_background_fill="linear-gradient(90deg, #f97316, #ea580c)",
    button_primary_background_fill_hover="linear-gradient(90deg, #ea580c, #c2410c)",
    button_primary_text_color="white",
    input_background_fill="#f8fafc",
    border_color_primary="#e5e7eb"
)


# ================================================================
# 🔹 UI Layout
# ================================================================
with gr.Blocks(theme=theme, css="""
#title {font-size: 2.2rem; font-weight: 700; text-align:center; margin-bottom: 0.3em; color:#1e293b;}
#subtitle {text-align:center; color:#64748b; margin-bottom: 1.5em; font-size:1.05em;}
#footer {text-align:center; font-size:0.85em; color:#94a3b8; margin-top:2em;}
""") as demo:

    # Header
    gr.HTML("<div id='title'>ScholarMind — Research Paper Summarizer</div>")
    gr.HTML("<div id='subtitle'>Upload your research PDF and generate a structured, evidence-backed summary.</div>")

    with gr.Row():
        with gr.Column(scale=1, min_width=320):
            pdf_input = gr.File(label="📄 Upload Research Paper (PDF)", file_types=[".pdf"])
            summary_level = gr.Radio(
                ["short", "detailed", "section"],
                value="short",
                label="Summary Type",
                info="Choose your preferred summary depth."
            )
            verify = gr.Checkbox(
                value=True,
                label="Enable factual verification",
                info="Checks claims using retrieved evidence."
            )
            submit_btn = gr.Button("Generate Summary", variant="primary", elem_id="submit-btn")
            clear_btn = gr.ClearButton([pdf_input], value="Clear")

        with gr.Column(scale=2):
            output = gr.Markdown(
                label="Summary Output",
                show_label=False,
                elem_id="output-box",
                value="👋 Upload a paper and click **Generate Summary** to begin.",
            )

    # Button actions
    submit_btn.click(
        summarize_pdf,
        inputs=[pdf_input, summary_level, verify],
        outputs=[output]
    )

    gr.HTML("<div id='footer'>© 2025 ScholarMind | AI-driven Research Understanding Platform</div>")


# ================================================================
# 🔹 Launch Gradio App
# ================================================================
if __name__ == "__main__":
    demo.launch(server_port=7860, server_name="127.0.0.1")
