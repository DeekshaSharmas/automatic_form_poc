import json
import os

SESSION_FILE = 'session_data.json'

def get_saved_url():
    
    """Reads the saved URL from session_data.json"""
    
    print(f"🔗 Reading saved URL from {SESSION_FILE}")
    if os.path.exists(SESSION_FILE):
        print(f"🔗 Reading saved URL from {SESSION_FILE}")
        with open(SESSION_FILE, 'r') as f:
            try:
                return json.load(f).get('url')
            except json.JSONDecodeError:
                print("❌ Invalid JSON in session file.")
    return None