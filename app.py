from flask import Flask, render_template, request, redirect, url_for
from flask import jsonify,session
import requests
import os
from concurrent.futures import ThreadPoolExecutor
import time
from flask_cors import CORS
import PyPDF2
from docx import Document
import hashlib
import secrets
import fitz  # PyMuPDF - install with: pip install PyMuPDF
import pdfplumber  # install with: pip install pdfplumber
import re
from html import escape

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = secrets.token_hex(16)
corsServers = '*'
CORS(app, resources={r"/*": {"origins": corsServers.split(",")}})
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Authentication credentials (change these to your desired username/password)
ADMIN_USERNAME = "kalidas"
ADMIN_PASSWORD = "admin123"

# Configuration
DEFAULT_MODELS = [
    'deepseek/deepseek-chat-v3-0324:free',
    'meta-llama/llama-3.3-70b-instruct:free'
]

def login_required(f):
    """Decorator to require login for certain routes"""
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

# Authentication Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '').strip()
    
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        session['logged_in'] = True
        session['username'] = username
        return jsonify({"status": "success", "message": "Login successful"})
    else:
        return jsonify({"status": "error", "message": "Invalid username or password"}), 401

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/check_login_status')
def check_login_status():
    return jsonify({"logged_in": session.get('logged_in', False)})

def get_model_info(model_id):
    """Get model display name from ID"""
    provider, model = model_id.split('/')
    return {
        'id': model_id,
        'name': f"{provider.capitalize()} {model.replace('-', ' ').title()}",
        'provider': provider
    }

def query_model(prompt: str, model_id: str) -> dict:
    """Query a single model through OpenRouter"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    #api_key = "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"
    if not api_key:
        return {
            "success": False,
            "model_id": model_id,
            "model_name": get_model_info(model_id)['name'],
            "error": "OpenRouter API key not set",
            "latency": 0
        }

    HEADERS = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://llm-ensemble.onrender.com",
        "Content-Type": "application/json",
        "X-Title": "LLM Ensemble Web App"
    }

    start_time = time.time()
    model_info = get_model_info(model_id)

    try:
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=HEADERS,
            json=payload,
            timeout=30
        )
        response.raise_for_status()

        return {
            "success": True,
            "model_id": model_id,
            "model_name": model_info['name'],
            "provider": model_info['provider'],
            "response": response.json()["choices"][0]["message"]["content"],
            "latency": round(time.time() - start_time, 2)
        }

    except Exception as e:
        return {
            "success": False,
            "model_id": model_id,
            "model_name": model_info['name'],
            "error": str(e),
            "latency": round(time.time() - start_time, 2)
        }

def consolidate_responses(prompt: str, responses: list) -> dict:
    """Use the best model to consolidate responses"""
    valid_responses = [r for r in responses if r["success"]]
    if not valid_responses:
        return {"error": "All model queries failed"}

    comparison = "\n\n".join(
        f"=== {r['model_name']} ===\n{r['response']}"
        for r in valid_responses
    )

    consolidation_prompt = f"""
    Analyze these responses to: "{prompt}"

    {comparison}

    Provide a structured comparison with no more than 300 words:
    1. Key points of agreement
    2. Major differences  
    3. Recommended synthesis
    4. Confidence level (High/Medium/Low)
    """

    # Use the first successful model for consolidation
    return query_model(consolidation_prompt, valid_responses[0]['model_id'])

# Web Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/llm_ensemble', methods=['GET', 'POST'])
def llm_ensemble():
    if request.method == 'GET':
        return render_template('llm_ensemble.html')

    # Handle form submission
    prompt = request.form.get('prompt', '').strip()
    if not prompt:
        return render_template('llm_ensemble.html', error="Please enter a question.")

    # Query all models in parallel
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(query_model, prompt, model_id) for model_id in DEFAULT_MODELS]
        results = [f.result() for f in futures]

    # Consolidate responses
    consolidation = consolidate_responses(prompt, results)

    return render_template('llm_ensemble.html', 
                         prompt=prompt,
                         results=results, 
                         consolidation=consolidation,
                         success_count=len([r for r in results if r['success']]),
                         total_count=len(results))

@app.route('/letters')
def letters():
    return render_template('letters.html')

@app.route('/upload_letter', methods=['POST'])
def upload_letter():
    try:
        password = request.form['password']
        document = request.files['document']
        force_update = request.form.get('force_update', 'false').lower() == 'true'
        
        print("Password:", password)
        print("Filename:", document.filename)
        print("Force update:", force_update)
        
        allowed_extensions = {'.pdf', '.docx'}
        file_ext = os.path.splitext(document.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            return jsonify({"status": "error", "message": "Only PDF and DOCX files are allowed"}), 400

        # Create directory based on password hash
        password_hash = hashlib.md5(password.encode()).hexdigest()
        directory_path = os.path.join("/Users/virinchiande/Desktop/LLM's", password_hash)
        #directory_path = os.path.join(UPLOAD_FOLDER,password_hash)
        
        # Check if directory already exists
        if os.path.exists(directory_path):
            if not force_update:
                # Return duplicate password status
                return jsonify({
                    "status": "duplicate_password", 
                    "message": "Duplicate password. The file will be updated with the new file"
                })
            else:
                # Delete existing files in the directory
                import shutil
                for filename in os.listdir(directory_path):
                    file_path = os.path.join(directory_path, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"Deleted existing file: {filename}")
        else:
            # Create new directory
            os.makedirs(directory_path, exist_ok=True)
        
        # Save the uploaded file
        document.save(os.path.join(directory_path, document.filename))
        print(f"Saved new file: {document.filename}")
        
        return jsonify({"status": "success", "message": "File uploaded successfully"})
        
    except Exception as e:
        print(f"Error in upload_letter: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Add this new route to your existing app.py
@app.route('/view_letters')
def view_letters():
    return render_template('view_letters.html')

def read_pdf(file_path):
    """Extract text from PDF file"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n\n"  # Double newlines for paragraphs
            return text.strip()
    except Exception as e:
        return f"Error reading PDF: {str(e)}"
    
def read_docx(file_path):
    """Extract text from DOCX file"""
    try:
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n\n"  # Double newlines for paragraphs
        return text.strip()
    except Exception as e:
        return f"Error reading DOCX: {str(e)}"
    
def format_text_for_display(text):
    """Convert plain text to HTML with proper paragraph breaks"""
    if not text:
        return ""

    # Split by double newlines to get paragraphs
    paragraphs = text.split('\n\n')
    formatted_paragraphs = []

    for para in paragraphs:
        para = para.strip()
        if para:
            # Replace single newlines with spaces within paragraphs
            para = para.replace('\n', ' ')
            formatted_paragraphs.append(f'<p class="document-paragraph">{para}</p>')

    return '\n'.join(formatted_paragraphs)

@app.route('/get_letter_content', methods=['POST'])
def get_letter_content():
    password = request.form.get('password', '').strip()

    if not password:
        return jsonify({"status": "error", "message": "Please enter a password"}), 400

    password_hash = hashlib.md5(password.encode()).hexdigest()
    upload_dir = f"/Users/virinchiande/Desktop/LLM's/{password_hash}"

    if not os.path.exists(upload_dir):
        return jsonify({"status": "error", "message": "Incorrect password"}), 404

    try:
        documents = []
        for filename in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, filename)

            if filename.lower().endswith('.pdf'):
                content = read_pdf(file_path)
                formatted_content = format_text_for_display(content)
                documents.append({
                    'filename': filename,
                    'type': 'PDF',
                    'content': formatted_content,
                    'raw_content': content
                })
            elif filename.lower().endswith('.docx'):
                content = read_docx(file_path)
                formatted_content = format_text_for_display(content)
                documents.append({
                    'filename': filename,
                    'type': 'DOCX',
                    'content': formatted_content,
                    'raw_content': content
                })

        if not documents:
            return jsonify({"status": "error", "message": "No documents found"}), 404

        return jsonify({
            "status": "success",
            "documents": documents
        })

    except Exception as e:
        return jsonify({"status": "error", "message": f"Error reading documents: {str(e)}"}), 500
    
if __name__ == '__main__':
    print("🚀 Starting Calcuttapuri LLM Ensemble Web App")
    print("📱 Main Website: http://localhost:5200/")
    print("🧠 LLM Ensemble: http://localhost:5200/llm_ensemble")
    print("⚠️  Make sure to set OPENROUTER_API_KEY environment variable")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5200)), debug=True)
