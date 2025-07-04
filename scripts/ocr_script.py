import os
import json
import re
import fitz  # PyMuPDF
from PIL import Image
import google.generativeai as genai
from flask.cli import load_dotenv
from docx2pdf import convert

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL")
if not api_key:
    raise EnvironmentError("GEMINI_API_KEY not set.")
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name=model_name)

OUTPUT_FILE = "output.json"

def append_any_json(json_string):
    try:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", json_string.strip(), flags=re.IGNORECASE | re.MULTILINE)
        new_data = json.loads(cleaned)
        if not isinstance(new_data, dict):
            raise ValueError("Only JSON objects (dicts) are allowed.")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"⚠️ Invalid JSON data: {e}")
        return

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = {}
            except json.JSONDecodeError:
                existing_data = {}
    else:
        existing_data = {}

    existing_data.update(new_data)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)

    print("✅ Data merged into output.json")

def process_image_with_gemini(img_path):
    try:
        img = Image.open(img_path)
        prompt = "Extract all relevant fields (name, DOB, Aadhaar, etc.) from this document image in JSON format."

        response = model.generate_content([prompt, img])
        print(f"📄 Gemini response for {img_path}:\n{response.text}")
        append_any_json(response.text)
    except Exception as e:
        print(f"❌ Error processing {img_path}: {e}")


def process_pdf(path):
    print(f"Processing PDF: {path}")
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return
    doc = fitz.open(path)
    for i, page in enumerate(doc):
        img_path = f"temp_page_{i+1}.png"
        pix = page.get_pixmap(dpi=150)
        pix.save(img_path)
        process_image_with_gemini(img_path)
        os.remove(img_path)


def process_docx(path):
    folder = os.path.dirname(path)
    base = os.path.splitext(os.path.basename(path))[0]
    output_pdf = os.path.join(folder, base + ".pdf")
    print(f"Processing DOCX: {path}")
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return
    # Convert DOCX to PDF then to images
    convert(path)
    process_pdf(output_pdf)
    # os.remove(output_pdf)


def process_image(path):
    print(f"Processing Image: {path}")
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return
    if not path.lower().endswith(('.jpg', '.jpeg', '.png')):
        print(f"⚠️ Unsupported image format: {path}")
        return
    process_image_with_gemini(path)
    
def process_document(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        process_pdf(file_path)
    elif ext == ".docx":
        process_docx(file_path)
    elif ext in [".jpg", ".jpeg", ".png"]:
        process_image(file_path)
    else:
        print(f"⚠️ Unsupported file format: {ext}")
        return f"❌ Unsupported file type: {ext}"
    if not os.path.exists(file_path):
        return "❌ File not found."

    doc = fitz.open(file_path)

    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=150)
        image_path = f"temp_page_{i + 1}.png"
        pix.save(image_path)

        img = Image.open(image_path)

        prompt = """
        You are an intelligent document parser. Extract all relevant fields and important details (like name, date of birth, gender, Aadhaar number, etc.)
        from the following page image and return the data in a JSON format. Structure the response as key-value pairs.
        """

        try:
            response = model.generate_content([prompt, img])
            print(f"\n📄 Page {i + 1} Response:\n", response.text)
            append_any_json(response.text)
        except Exception as e:
            print(f"⚠️ Error processing page {i + 1}: {e}")

        os.remove(image_path)

    return "✅ Document processed successfully."
