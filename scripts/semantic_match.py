import json
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# === Load fine-tuned model globally (for reuse) ===
model_path = "./t5-semantic-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

def perform_semantic_mapping():
    """
    Reads form fields from scraped HTML and user data from output.json,
    maps them using a fine-tuned T5 model, and saves the result as semantic_mapping.json.

    Returns:
        dict: A dictionary with form field names mapped to user data values.
    """

    # === Hardcoded file paths ===
    html_file_path = "scraped_page.html"
    output_json_path = "output.json"
    output_mapping_path = "semantic_mapping.json"

    def semantic_map(input_key: str) -> str:
        inputs = tokenizer(input_key, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=10,
            num_beams=4,
            early_stopping=True
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # === Load user data ===
    with open(output_json_path, "r", encoding="utf-8") as f:
        user_data = json.load(f)

    # === Load and parse HTML form ===
    with open(html_file_path, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    form_fields = {}

    for tag in soup.find_all(["input", "textarea", "select"]):
        name = tag.get("name")
        if name:
            form_fields[name] = ""

    # === Perform semantic mapping ===
    final_mapping = {}

    for form_key in form_fields.keys():
        predicted_user_key = semantic_map(form_key)
        user_value = user_data.get(predicted_user_key, "")
        final_mapping[form_key] = user_value
        print(f"🔁 {form_key} → {predicted_user_key} → {user_value}")

    # === Save mapping to JSON file ===
    with open(output_mapping_path, "w", encoding="utf-8") as f:
        json.dump(final_mapping, f, indent=2)

    print(f"\n✅ Semantic mapping saved to {output_mapping_path}")
    return True, "Semantic mapping generated successfully."
