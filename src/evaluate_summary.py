"""
evaluate_summaries.py
--------------------------------------
Evaluate ScholarMind summarization outputs using ROUGE, BLEU, and (optional) BERTScore.

Expected input (JSONL or CSV) can include any of the following keys:
{
    "prediction" | "pred" | "generated_text" | "model_output",
    "reference"  | "target_text" | "summary" | "gold"
}

Usage:
    python evaluate_summaries.py --file data/finetune/val.jsonl --use_bertscore
"""

import json
import argparse
from tqdm import tqdm
from evaluate import load
import os

# ============================================================
# 🔹 Argument Parser
# ============================================================
parser = argparse.ArgumentParser(description="Evaluate model summaries using ROUGE, BLEU, and BERTScore")
parser.add_argument("--file", type=str, required=True, help="Path to JSONL or CSV file with reference/prediction pairs")
parser.add_argument("--use_bertscore", action="store_true", help="Include BERTScore in evaluation")
args = parser.parse_args()

# ============================================================
# 🔹 Load Data
# ============================================================
print(f"\n📘 Loading data from: {args.file}")

data = []
if not os.path.exists(args.file):
    raise FileNotFoundError(f"File not found: {args.file}")

if args.file.endswith(".jsonl"):
    with open(args.file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
elif args.file.endswith(".csv"):
    import pandas as pd
    df = pd.read_csv(args.file)
    data = df.to_dict("records")
else:
    raise ValueError("Input file must be .jsonl or .csv")

# ============================================================
# 🔹 Extract Predictions and References
# ============================================================
preds, refs = [], []
for item in tqdm(data, desc="Parsing samples"):
    pred = (
        item.get("prediction")
        or item.get("pred")
        or item.get("generated_text")
        or item.get("model_output")
        or item.get("input_text")  # fallback if dataset uses 'input_text' as model output
    )
    ref = (
        item.get("reference")
        or item.get("target_text")
        or item.get("summary")
        or item.get("gold")
        or item.get("target")
    )

    if pred and ref:
        preds.append(str(pred).strip())
        refs.append(str(ref).strip())

if not preds or not refs:
    raise ValueError(
        "❌ No valid prediction/reference pairs found.\n"
        "Ensure your file contains keys like 'prediction'/'reference' or 'target_text'."
    )

print(f"✅ Loaded {len(preds)} valid pairs for evaluation.")

# ============================================================
# 🔹 Load Metrics
# ============================================================
print("\n🔹 Loading evaluation metrics...")
rouge = load("rouge")
bleu = load("bleu")
results = {}

# ============================================================
# 🔹 Compute ROUGE
# ============================================================
print("🔹 Computing ROUGE...")
rouge_scores = rouge.compute(predictions=preds, references=refs)
results["ROUGE"] = {k: round(v * 100, 2) for k, v in rouge_scores.items()}

# ============================================================
# 🔹 Compute BLEU
# ============================================================
print("🔹 Computing BLEU...")
bleu_scores = bleu.compute(predictions=preds, references=refs)
results["BLEU"] = round(bleu_scores["bleu"] * 100, 2)

# ============================================================
# 🔹 Compute BERTScore (optional)
# ============================================================
if args.use_bertscore:
    print("🔹 Computing BERTScore... (this may take a few minutes on first run)")
    bertscore = load("bertscore")
    bert_scores = bertscore.compute(predictions=preds, references=refs, lang="en")
    results["BERTScore"] = {
        "precision": round(sum(bert_scores["precision"]) / len(bert_scores["precision"]) * 100, 2),
        "recall": round(sum(bert_scores["recall"]) / len(bert_scores["recall"]) * 100, 2),
        "f1": round(sum(bert_scores["f1"]) / len(bert_scores["f1"]) * 100, 2),
    }

# ============================================================
# 🔹 Display & Save Results
# ============================================================
print("\n✅ Final Evaluation Results:")
for metric, score in results.items():
    print(f"{metric}: {score}")

os.makedirs("results", exist_ok=True)
output_path = os.path.join("results", "evaluation_results.json")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\n📁 Saved evaluation metrics to: {output_path}")
