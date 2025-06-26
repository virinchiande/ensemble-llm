from flask import Flask, render_template, request, jsonify
import requests
import os
from concurrent.futures import ThreadPoolExecutor
import time
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
corsServers = '*'
CORS(app, resources={r"/*": {"origins": corsServers.split(",")}})

# Configuration
DEFAULT_MODELS = [
    'deepseek/deepseek-chat-v3-0324:free',
    'meta-llama/llama-3.3-70b-instruct:free'
]

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

if __name__ == '__main__':
    print("🚀 Starting Calcuttapuri LLM Ensemble Web App")
    print("📱 Main Website: http://localhost:5200/")
    print("🧠 LLM Ensemble: http://localhost:5200/llm_ensemble")
    print("⚠️  Make sure to set OPENROUTER_API_KEY environment variable")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
