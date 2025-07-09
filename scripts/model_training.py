import json
import sys
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

# === CONFIGURATION ===
MODEL_NAME = "ft:gpt-3.5-turbo-0125:personal::BYzLBlz4"  # or any small open-source causal model
OUTPUT_DIR = "./fine_tuned_model"
MAX_LENGTH = 512
EPOCHS = 3
BATCH_SIZE = 2

# === STEP 1: LOAD & FORMAT JSON ===
def format_record(record):
    values = record["extracted_values"]
    prompt = "### Instruction:\nFill this form based on the given details.\n\n"
    for key, value in values.items():
        prompt += f"{key}: {value}\n"
    completion = f"### Response:\nForm generated for applicant {values['applicantName']} under {values['complaintNature']}."
    return {"text": f"{prompt}\n{completion}"}

def load_and_prepare_dataset(json_path):
    with open(json_path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    formatted = [format_record(item) for item in data]
    return Dataset.from_list(formatted)

# === STEP 2: TOKENIZE ===
def tokenize_dataset(dataset, tokenizer):
    def tokenize_fn(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
    return dataset.map(tokenize_fn, batched=True)

# === STEP 3: TRAIN ===
def fine_tune(dataset):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        logging_steps=10,
        save_steps=50,
        learning_rate=5e-5,
        evaluation_strategy="no",
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

# === MAIN ===
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fine_tune_form_model.py path_to_your_json_file.json")
        sys.exit(1)

    json_file = sys.argv[1]
    dataset = load_and_prepare_dataset(json_file)
    fine_tune(dataset)
    print(f"Model fine-tuned and saved to {OUTPUT_DIR}")
