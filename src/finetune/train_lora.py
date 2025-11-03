import os
os.environ["WANDB_DISABLED"] = "true"

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

MODEL_NAME = 'facebook/bart-large-cnn'  # change as needed


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=['q_proj','v_proj'] if hasattr(model, 'get_input_embeddings') else ['q','v'],
        lora_dropout=0.1,
        bias='none'
    )
    model = get_peft_model(model, lora_config)

    import os

    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/finetune"))
    
    train_path = os.path.join(base_path, "train.jsonl")
    val_path = os.path.join(base_path, "val.jsonl")
    
    print("Using dataset paths:")
    print(train_path)
    print(val_path)
    
    ds = load_dataset('json', data_files={'train': train_path, 'validation': val_path})
    def preprocess(batch):
        inp = tokenizer(batch['input_text'], truncation=True, padding='max_length', max_length=1024)
        targ = tokenizer(batch['target_text'], truncation=True, padding='max_length', max_length=256)
        inp['labels'] = targ['input_ids']
        return inp
    ds = ds.map(preprocess, batched=True, remove_columns=ds['train'].column_names)

    training_args = Seq2SeqTrainingArguments(
        output_dir='checkpoints/lora',
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        predict_with_generate=True,
        logging_steps=50,
        eval_strategy='steps',
        eval_steps=500,
        save_steps=1000,
        num_train_epochs=1,
        fp16=True,
        push_to_hub=False,
        gradient_accumulation_steps=4,
    )
    trainer = Seq2SeqTrainer(model=model, tokenizer=tokenizer, args=training_args, train_dataset=ds['train'], eval_dataset=ds['validation'])
    trainer.train()

if __name__ == '__main__':
    main()