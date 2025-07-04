import ast
import json
import os
import re
import openai
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("CHATGPT_API_KEY")
fine_tune_model = os.getenv("FINE_TUNE_MODEL")
if not api_key:
    raise EnvironmentError("CHATGPT_API_KEY not set.")

# ✅ Load API key safely
openai.api_key = api_key
def generate_semantic_mapping():
    try:
        # ✅ Load user data (extracted values like name, dob, etc.)
        with open("output.json", "r", encoding="utf-8") as file:
            form_data = json.load(file)

        # ✅ Load scraped HTML form
        with open("scraped_page.html", "r", encoding="utf-8") as file:
            html_content = file.read()

        # ✅ Parse HTML form and extract fields
        soup = BeautifulSoup(html_content, 'html.parser')
        form_fields = {}

        for tag in soup.find_all(['input', 'textarea', 'select']):
            name = tag.get('name')
            if name:
                form_fields[name] = ""  # Store form fields as key names

        # ✅ Save extracted form fields
        with open("form_fields.json", "w", encoding="utf-8") as f:
            json.dump(form_fields, f, indent=2)

        print("✅ Form fields saved to form_fields.json")

        # ✅ Ask OpenAI to semantically match form fields to user data
        prompt = f"""
        Match each form field key from this dict: {form_fields}
        to the most appropriate key-value pair from this user data: {form_data}.
        Return the output as a valid JSON dictionary where:
        - keys = form field names
        - values = selected values from the user data
        """

        response = openai.chat.completions.create(
            model=fine_tune_model,  # Replace with your actual fine-tuned model ID
            messages=[{"role": "user", "content": prompt}]
        )

        ai_response = response.choices[0].message.content
        print("🔁 Predicted Mapping:\n", ai_response)
       

        # ✅ Try parsing the response
        try:
            predicted_mapping = json.loads(ai_response)
        except json.JSONDecodeError:
            predicted_mapping = ast.literal_eval(ai_response)
            
        
        # ✅ Save the predicted mapping
        with open("semantic_mapping.json", "w", encoding="utf-8") as f:
            json.dump(predicted_mapping, f, indent=2)

        print("✅ Semantic mapping saved to semantic_mapping.json")
        return True, "Semantic mapping generated successfully."

    except Exception as e:
        print("❌ Error in semantic mapping:", e)
        return False, f"Error generating semantic mapping: {str(e)}"
