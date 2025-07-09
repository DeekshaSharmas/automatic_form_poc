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
# Optimized Selenium Form Automation Prompt

## Input Data
- **Semantic Mapping JSON**: `{mapped_json}` - Dictionary where keys are form field identifiers and values are the actual data to fill in those fields
- **HTML Content**: `{html_content}` - Complete HTML source of the target form
- **Target URL**: `{url}` - URL where the form is located

## Critical Requirements
Generate a complete Python Selenium script that:

### 1. Setup and Navigation
```python
# Import: selenium, json, time, WebDriverWait, By, Select, Keys
# Setup Chrome WebDriver (headless=False, maximize window)
# Navigate to {url}
# Wait for page load with WebDriverWait
```

### 2. Form Filling Logic - MOST IMPORTANT
**The script MUST iterate through each key-value pair in `{mapped_json}` and fill the corresponding form fields:**

For each item in `{mapped_json}`:
- **Key** = field identifier (name, id, or selector)
- **Value** = data to fill in that field

**Optimized Field Location Strategy (try in order):**
1. `driver.find_element(By.ID, key)`
2. `driver.find_element(By.NAME, key)`
3. `driver.find_element(By.ID, key.lower())`
4. `driver.find_element(By.NAME, key.lower())`
5. `driver.find_element(By.CSS_SELECTOR, f'[name*="key"]')`

**Fill Logic by Field Type:**
- **Text/Textarea**: `element.clear()` then `element.send_keys(value)`
- **Select Dropdown**: Try `Select(element).select_by_visible_text(value)` then `Select(element).select_by_value(value)`
- **Checkbox**: Check if value is truthy, then `element.click()` if not already checked
- **Radio Button**: Find radio with matching value and click
- **File Upload**: `element.send_keys(file_path)` if value is file path

### 3. Optimized Monitoring Setup
**Set up lightweight monitoring ONLY after form filling:**
- Add single event listener to form: `form.addEventListener('submit', function()  window.formSubmitted = true; )`
- Monitor URL changes by storing initial URL
- **DO NOT add listeners to individual form fields** (this causes slowness)
- **DO NOT extract values continuously** (only on submit or URL change)

### 4. Efficient Detection Methods
**Priority Detection (implement in order):**

1. **Form Submit Detection**: Monitor `window.formSubmitted` flag
2. **URL Change Detection**: Compare `driver.current_url` with initial URL
3. **Page Title Change**: Monitor `driver.title` changes
4. **Alert Detection**: Check for alerts only when needed

**Value Extraction Strategy:**
- Extract values ONLY when:
  - Form submission detected
  - URL changes
  - Manual user request (keyboard shortcut)
- **DO NOT extract on every click/change** (causes slowness)

### 5. Optimized Monitoring Loop
```python
# Efficient monitoring pattern:
def monitor_form_efficiently(driver, mapped_json):
    initial_url = driver.current_url
    initial_title = driver.title
    
    print("Form filling completed. Monitoring for submission...")
    print("Press Ctrl+E to extract values manually, Ctrl+C to exit")
    
    while True:
        try:
            # Check form submission (lightweight)
            if driver.execute_script("return window.formSubmitted || false"):
                print("🚀 Form submission detected!")
                extract_filled_values(driver, mapped_json)
                driver.execute_script("window.formSubmitted = false")
            
            # Check URL change (lightweight)
            elif driver.current_url != initial_url:
                print(f"🔄 URL changed to: driver.current_url")
                extract_filled_values(driver, mapped_json)
                initial_url = driver.current_url
            
            # Check title change (lightweight)
            elif driver.title != initial_title:
                print(f"📄 Page title changed to: driver.title")
                extract_filled_values(driver, mapped_json)
                initial_title = driver.title
            
            time.sleep(2)  # Check every 2 seconds (not 0.5 seconds)
            
        except KeyboardInterrupt:
            print("Monitoring stopped by user")
            break
        except Exception as ex:
            print(f"Monitoring error: ex")
            time.sleep(1)
```

### 6. Streamlined Value Extraction with JSON Saving
```python
def extract_filled_values(driver, mapped_json):
    extracted_values = blank
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    for field_key in mapped_json.keys():
        try:
            element = locate_element_efficiently(driver, field_key)
            if element:
                value = get_element_value(element)
                if value:
                    extracted_values[field_key] = value
        except:
            continue
    
    # Add timestamp to extracted data
    extracted_data = 
        "timestamp": timestamp,
        "extracted_values": extracted_values
    
    
    # Save to JSON file
    try:
        with open("extracted.json", "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Extracted values saved to extracted.json")
    except Exception as ex:
        print(f"❌ Error saving to JSON: ex")
    
    print(f"\n=== Extracted Values at timestamp ===")
    print(json.dumps(extracted_values, indent=2))
    print("=" * 50)
```

### 7. Performance Optimizations
- **Reduce polling frequency**: Check every 2 seconds instead of 0.5 seconds
- **Minimal JavaScript**: Use simple flags instead of complex event listeners
- **Lazy extraction**: Only extract values when necessary
- **Efficient selectors**: Use direct ID/name lookups first
- **No continuous monitoring**: Avoid real-time field change detection
- **Batch operations**: Process multiple fields efficiently

### 8. Error Handling Pattern
```python
for field_key, field_value in mapped_json.items():
    try:
        element = locate_element_efficiently(driver, field_key)
        if element:
            fill_element_efficiently(element, field_value)
            print(f"✓ Filled field_key")
    except Exception as ex:
        print(f"✗ Failed to fill field_key: str(ex)")
        continue
```

### 9. Output Requirements
- Return ONLY the complete Python script
- No explanations or comments outside code
- Script must be immediately executable
- Include all necessary imports
- **Focus on performance and minimal resource usage**

### 10. Critical Performance Rules
- **NO individual field event listeners**
- **NO continuous value extraction**
- **NO frequent polling (max every 2 seconds)**
- **NO complex DOM monitoring**
- **YES to simple form submit detection**
- **YES to URL change monitoring**
- **YES to efficient element location**
- **YES to saving extracted values in extracted.json file**

**The script must be FAST and EFFICIENT while still detecting form submissions reliably and saving extracted values to extracted.json.**
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
        # try:
        #     result = subprocess.run(["python", "generated_fill_form2.PY"], capture_output=True, text=True, check=True)

        #     print("📄 Script Output:\n", result.stdout)
        #     if result.stderr:
        #         print("⚠️ Script Errors:\n", result.stderr)
        # except subprocess.CalledProcessError as e:
        #     print("❌ Error executing script:", e)
        #     return False, f"❌ Failed to execute script: {str(e)}"

        return True, "✅ Script generated and executed successfully."
        # return True, "✅ Form filler script generated successfully."
    
    except Exception as e:
        print("❌ Error generating form filler script:", e)
        return False, f"❌ Failed to generate form filler script: {str(e)}"
