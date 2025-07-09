import json
import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from scripts.dynamic_script import generate_dynamic_form_script
from scripts.ocr_script import process_document, append_any_json
from scripts.saveurl_script import get_saved_url
from scripts.scrape_form import scrape_saved_url
from scripts.semnatic_mapping import generate_semantic_mapping
from scripts.semantic_match import perform_semantic_mapping
from scripts.retrain_model import train_model

app = Flask(__name__)
SESSION_FILE = "session_data.json"

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print(f"Uploading file: {filename} to {file_path}")
    file.save(file_path)
    return jsonify({'message': 'File uploaded successfully', 'path': file_path})

@app.route('/run_ocr', methods=['GET'])
def run_ocr():
    upload_dir = app.config['UPLOAD_FOLDER']
    files = [f for f in os.listdir(upload_dir) if f.lower().endswith(('.pdf', '.docx', '.png', '.jpg'))]

    if not files:
        return jsonify({'error': 'No uploaded files found'}), 400
    results = []
    print("Processing the following files:",files)

    for file in files:
        filename = secure_filename(file)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"Processing file inside app.py: {filename} at {file_path}")
        result = process_document(file_path)  # ✅ Your OCR logic here
        results.append({filename: result})

    return jsonify({'message': 'All documents processed.', 'details': results})
    # try:
        
    #     result = process_document()
    #     append_any_json(result)
    #     return jsonify({'message': 'OCR processing completed', 'result': result})
    # except Exception as e:
    #     return jsonify({'error': str(e)}), 500
@app.route('/save_url', methods=['POST'])
def save_url():
    data = request.get_json()
    url = data.get('url')
    print(f"🔗 Received URL: {url}")
    # Save the URL to session_data.json
    if url:
        session_data = {}
        print(f"🔗 Reading saved URL from {SESSION_FILE}")
        if os.path.exists(SESSION_FILE):
            with open(SESSION_FILE, 'r') as f:
                try:
                    session_data = json.load(f)
                except json.JSONDecodeError:
                    pass
        
        session_data['url'] = url
        with open(SESSION_FILE, 'w') as f:
            json.dump(session_data, f, indent=4)

        print(f"🔗 Saved URL for later use: {url}")

    # You can now run your OCR logic here
    return jsonify({'message': 'OCR executed and URL saved successfully'})

@app.route('/fetch_url', methods=['GET'])
def fetch_url():
    url = get_saved_url()
    print(f"🔗 Fetching saved URL: {url}")
    if not url:
        return jsonify({'error': '❌ No saved URL found'}), 404

    return jsonify({'url': url, 'message': '✅ Saved URL fetched successfully'})

# @app.route('/scrape_url', methods=['GET'])
# def scrape_url():
#     success, message = scrape_saved_url()
    
#     print(f"🔗 Scraping URL: {message}")
#     if success:
#         mapping_success, mapping_message = perform_semantic_mapping()
        
#         print(f"🔗 Semantic mapping: {mapping_message}")
#         if not mapping_success:
#             return jsonify({'error': f"Mapping failed: {mapping_message}"}), 500
        
#         form_script_success, form_msg = generate_dynamic_form_script()
#         if not form_script_success:
#             return jsonify({'error': f"Script generation failed: {form_msg}"}), 500

#         return jsonify({'message': "✅ Scrape, mapping, and script generation completed."})

#         # return jsonify({'message': '✅ Scraping and semantic mapping completed successfully.'})
#     else:
#         return jsonify({'error': message}), 500
    
@app.route('/scrape_url', methods=['GET'])
def scrape_url():
    success, message = scrape_saved_url()
    print(f"🔗 Scraping URL: {message}")

    if success:
        mapping_success, mapping_message = perform_semantic_mapping()
        print(f"🔗 Semantic mapping: {mapping_message}")

        if not mapping_success:
            return jsonify({'error': f"Mapping failed: {mapping_message}"}), 500

        form_script_success, form_msg = generate_dynamic_form_script()
        print(f"🧠 Script Generation: {form_msg}")

        if not form_script_success:
            return jsonify({'error': f"Script generation failed: {form_msg}"}), 500

        # ✅ Now run the generated script
        try:
            import subprocess

            print("🚀 Running generated script...")
            result = subprocess.run(
                ["python", "generated_fill_form2.py"],  # ensure lowercase `.py`
                capture_output=True,
                text=True,
                check=True
            )

            print("📄 Script Output:\n", result.stdout)
            if result.stderr:
                print("⚠️ Script Errors:\n", result.stderr)
                
            train_model()
            return jsonify({
                'message': "✅ Scrape, mapping, script generation, and script execution completed.",
                'script_output': result.stdout
            })

        except subprocess.CalledProcessError as e:
            print("❌ Script execution failed:", e)
            return jsonify({
                'error': f"Script execution failed: {e}",
                'stderr': e.stderr
            }), 500

    else:
        return jsonify({'error': message}), 500

if __name__ == '__main__':
    app.run(debug=True)
