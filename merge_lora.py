from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
import os

base_model_name = "facebook/bart-large-cnn"
adapter_dir = "checkpoints/lora/checkpoint-250"  # path where LoRA adapter is saved
merged_dir = "checkpoints/merged_model/checkpoint-250"  # folder to save final merged model

# Load base model + LoRA adapter
model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(model, adapter_dir)

# Merge LoRA weights into the base model
model = model.merge_and_unload()

# Save merged model
os.makedirs(merged_dir, exist_ok=True)
model.save_pretrained(merged_dir)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.save_pretrained(merged_dir)

print(f"✅ Model successfully merged and saved at: {merged_dir}")
