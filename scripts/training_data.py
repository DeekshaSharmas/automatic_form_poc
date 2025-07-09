import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset

# === Load and preprocess your JSON data ===
with open('data.json') as f:
    raw_data = json.load(f)

# Convert to Hugging Face Dataset
dataset = Dataset.from_list(raw_data)

# dataset = load_dataset("json", data_files="train.jsonl", split="train")

# === Load T5 tokenizer and model ===
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# === Tokenize the dataset ===
def preprocess(example):
    input_enc = tokenizer(example["input_text"], truncation=True, padding="max_length", max_length=32)
    target_enc = tokenizer(example["target_text"], truncation=True, padding="max_length", max_length=16)
    input_enc["labels"] = target_enc["input_ids"]
    return input_enc

# def preprocess(example):
#     input_enc = tokenizer(example["input"], truncation=True, padding="max_length", max_length=32)
#     target_enc = tokenizer(example["output"], truncation=True, padding="max_length", max_length=16)
#     input_enc["labels"] = target_enc["input_ids"]
#     return input_enc


tokenized_dataset = dataset.map(preprocess)

# === Define training arguments ===
training_args = TrainingArguments(
    output_dir="./t5-semantic-model",
    per_device_train_batch_size=4,
    num_train_epochs=10,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=50,
    save_total_limit=2
)


# === Create trainer ===
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# === Start training ===
trainer.train()

# === Save model ===
model.save_pretrained("./t5-semantic-model")
tokenizer.save_pretrained("./t5-semantic-model")
