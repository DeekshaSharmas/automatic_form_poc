import json
from selenium.webdriver.common.by import By

def extract_filled_values(driver, mapped_json):
    
    extracted_data = {}

    for field, json_key in mapped_json.items():
        try:
            elements = driver.find_elements(By.NAME, field)
            if not elements:
                elements = driver.find_elements(By.ID, field)

            if not elements:
                print(f"[!] Field '{field}' not found.")
                continue

            element = elements[0]
            tag = element.tag_name.lower()
            type_attr = element.get_attribute("type")

            if tag == "input" and type_attr in ["text", "email", "password", "number"]:
                extracted_data[json_key] = element.get_attribute("value")

            elif tag == "textarea":
                extracted_data[json_key] = element.get_attribute("value")

            elif tag == "select":
                from selenium.webdriver.support.ui import Select
                select = Select(element)
                extracted_data[json_key] = select.first_selected_option.get_attribute("value")

            elif tag == "input" and type_attr == "checkbox":
                checkboxes = driver.find_elements(By.NAME, field)
                extracted_data[json_key] = [cb.get_attribute("value") for cb in checkboxes if cb.is_selected()]

            elif tag == "input" and type_attr == "radio":
                radios = driver.find_elements(By.NAME, field)
                selected = next((rb.get_attribute("value") for rb in radios if rb.is_selected()), None)
                extracted_data[json_key] = selected

            else:
                extracted_data[json_key] = element.get_attribute("value")

        except Exception as e:
            print(f"Error reading '{field}':", str(e))

    print("\n📋 Extracted Data from Form:")
    print(json.dumps(extracted_data, indent=2))

    with open("extracted_values.json", "w") as f:
        json.dump(extracted_data, f, indent=2)
