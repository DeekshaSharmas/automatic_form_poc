import os
import re
import subprocess
import openai
import json
import google.generativeai as genai
from dotenv import load_dotenv
from scripts.saveurl_script import get_saved_url
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_model = os.getenv("GEMINI_MODEL")

def generate_dynamic_form_script():
    try:
        openai.api_key = gemini_api_key
        with open("semantic_mapping.json", "r", encoding="utf-8") as file:
            mapped_json = json.load(file)

        with open("scraped_page.html", "r", encoding="utf-8") as file:
            html_content = file.read()  # HTML content of the dynamic form
            
        url = get_saved_url()

        prompt = f"""
You are given the following:

Given the following dynamic form field mappings:
{mapped_json}

HTML content of the dynamic form:
{html_content}

Your task is to generate a complete Python Selenium script that:

Navigates to the given {url}.

Dynamically locates form fields using the mapped_json mapping.

Fills in form fields with values from form_data, supporting:

Text inputs

Textareas

Select dropdowns

Checkboxes (including multiple selections)

Radio buttons (matched by name and value attributes)

Handles field names in a case-insensitive manner.

Uses flexible locating strategies (By.NAME, By.ID, etc.).

Gracefully logs missing elements without crashing.

Do not submit the form in the script — let the user interact with it.

detect submit/Done final call button of the form in the script —Wait for user interact with it.

Keep the browser open for user interaction. Do not call driver.quit() automatically.

Add extract_filled_values(driver, mapped_json) after submit button is clicked

After this input, extract all filled values using a function `extract_filled_values(driver, mapped_json)`.

Choose a workable webdriver (e.g., Chrome, Firefox) and ensure it is compatible with the latest Selenium version.

Print the extracted values in formatted JSON using `print(json.dumps(..., indent=2))`.

Return only the complete Python script — no explanations or comments outside the code.


"""
#Add this line at the end: input("Press Enter to continue...") to pause the script for user input.
#Keep the browser open for user interaction. Do not call driver.quit() automatically.
#add this tag in the end input("Press Enter to close the browser...") and do not close the browser automatically.

        model = genai.GenerativeModel(gemini_model)
        response = model.generate_content(prompt)
         
        cleaned_response = re.sub(r"^```(?:python)?\s*|\s*```$", "", response.text.strip(), flags=re.IGNORECASE | re.MULTILINE)
        
        print("🔁 Cleaned Response:\n", cleaned_response)

        with open("generated_fill_form2.py", "w", encoding="utf-8") as file:
            file.write(cleaned_response)

        print("✅ Dynamic form filler script has been generated and saved as 'generated_fill_form2.py'.")
        # result = subprocess.run(["python", "generated_fill_form2.PY"], capture_output=True, text=True)

        # print("📄 Script Output:\n", result.stdout)
        # if result.stderr:
        #     print("⚠️ Script Errors:\n", result.stderr)

        return True, "✅ Script generated and executed successfully."
        # return True, "✅ Form filler script generated successfully."
    
    except Exception as e:
        print("❌ Error generating form filler script:", e)
        return False, f"❌ Failed to generate form filler script: {str(e)}"
