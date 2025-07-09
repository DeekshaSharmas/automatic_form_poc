import json
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
import torch
from sklearn.model_selection import train_test_split
import numpy as np

# === Device Setup ===
def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        print("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device

device = setup_device()

# === Config ===
MODEL_NAME = "t5-small"
OUTPUT_DIR = "./t5-semantic-model"
MAX_INPUT_LENGTH = 64
MAX_TARGET_LENGTH = 32
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 5e-4
WARMUP_STEPS = 100

# === Dataset ===
dataset = load_dataset("json", data_files="train.jsonl", split="train")

def split_dataset(dataset, test_size=0.2, random_state=42):
    train_indices, val_indices = train_test_split(
        range(len(dataset)), test_size=test_size, random_state=random_state
    )
    return dataset.select(train_indices), dataset.select(val_indices)

train_dataset, val_dataset = split_dataset(dataset)

# === Tokenizer and Model ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.to(device)

# Add custom special tokens
special_tokens = {
    "additional_special_tokens": [
        "<map>", "</map>", "<field>", "</field>", "<value>", "</value>"
    ]
}
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))

# === Preprocess Function ===
def preprocess(example):
    input_text = f"map field: {example['input']}"
    target_text = example['output']
    input_enc = tokenizer(input_text, truncation=True, padding="max_length", max_length=MAX_INPUT_LENGTH, return_tensors="pt")
    target_enc = tokenizer(target_text, truncation=True, padding="max_length", max_length=MAX_TARGET_LENGTH, return_tensors="pt")
    labels = target_enc["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return {
        "input_ids": input_enc["input_ids"].flatten(),
        "attention_mask": input_enc["attention_mask"].flatten(),
        "labels": labels.flatten()
    }

# === Tokenize ===
tokenized_train = train_dataset.map(preprocess, remove_columns=train_dataset.column_names)
tokenized_val = val_dataset.map(preprocess, remove_columns=val_dataset.column_names)

# === TrainingArguments ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_steps=50,
    save_steps=50,
    save_total_limit=3,
    eval_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=False,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to=None,
)

# === Metrics ===
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predicted_ids = np.argmax(predictions, axis=-1)
    decoded_preds = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    exact_matches = sum(pred.strip() == label.strip() for pred, label in zip(decoded_preds, decoded_labels))
    accuracy = exact_matches / len(decoded_preds) if decoded_preds else 0
    return {
        "accuracy": accuracy,
        "exact_matches": exact_matches,
        "total_samples": len(decoded_preds)
    }

# === Data Collator ===
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, return_tensors="pt")

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# === Exposed Functions ===
def train_model():
    try:
        print("Starting training...")
        trainer.train()
        print("Evaluating final model...")
        eval_results = trainer.evaluate()
        print(f"Final evaluation results: {eval_results}")
        print("Saving model and tokenizer...")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        with open(os.path.join(OUTPUT_DIR, "training_config.json"), 'w') as f:
            json.dump({
                "model_name": MODEL_NAME,
                "max_input_length": MAX_INPUT_LENGTH,
                "max_target_length": MAX_TARGET_LENGTH,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "final_eval_results": eval_results
            }, f, indent=2)

        print(f"Training complete. Model saved to {OUTPUT_DIR}")
        return eval_results
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise

def test_model_inference():
    model.to(device)
    model.eval()
    test_inputs = ["language", "ministry", "applicantName", "email", "pincode"]
    results = {}
    for test_input in test_inputs:
        input_text = f"map field: {test_input}"
        inputs = tokenizer(input_text, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        try:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=MAX_TARGET_LENGTH,
                    num_beams=3,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            prediction = tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
            results[test_input] = prediction
        except Exception as e:
            results[test_input] = f"Error: {str(e)}"
    return results
