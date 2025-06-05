# openrouter_swagger_api.py
from flask import Flask
from flask_restx import Api, Resource, fields
import requests
import os
from concurrent.futures import ThreadPoolExecutor
import time

# Initialize Flask app
app = Flask(__name__)
api = Api(
    app,
    version='1.0',
    title='LLM Ensemble API',
    description='Query multiple LLMs through OpenRouter with Swagger UI',
    doc='/swagger-ui'
)

# Namespace for our operations
ns = api.namespace('ensemble', description='LLM operations')

# Model for Swagger documentation
ensemble_model = api.model('EnsembleRequest', {
    'prompt': fields.String(required=True, description='The input prompt for LLMs', example='Explain quantum computing'),
    #'models': fields.List(
        #fields.String,
        #description='List of model IDs to query (optional)',
        #example=['openai/gpt-3.5-turbo', 'anthropic/claude-3-sonnet'],
        #default=[]
    #)
})

# Configuration
OPENROUTER_API_KEY = "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"
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://llm-ensemble.onrender.com",
    "X-Title": "LLM Ensemble API"
}

# Default models (OpenRouter model IDs)
DEFAULT_MODELS = [
    'deepseek/deepseek-chat-v3-0324:free',
    'qwen/qwen3-30b-a3b:free',
    'google/gemini-2.0-flash-exp:free',
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
            timeout=15
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
    
    # Use Claude for consolidation (better for analysis)
    return query_model(consolidation_prompt, 'deepseek/deepseek-chat-v3-0324:free')

@ns.route('/testApi')
class testApi(Resource):
    def get(self):
        return {"status": "API is running!"}

@ns.route('/query')
class EnsembleQuery(Resource):
    @ns.expect(ensemble_model)
    @ns.response(200, 'Success')
    @ns.response(400, 'Validation Error')
    def post(self):
        """Query multiple LLMs with the same prompt"""
        data = api.payload
        prompt = data.get('prompt')
        #model_ids = data.get('models', DEFAULT_MODELS)
        
        if not prompt:
            return {"error": "Prompt is required"}, 400
        
        # Query all models in parallel
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(query_model, prompt, model_id) for model_id in DEFAULT_MODELS]
            results = [f.result() for f in futures]
        
        # Consolidate responses
        consolidation = consolidate_responses(prompt, results)
        
        return {
            "prompt": prompt,
            "models_queried": [get_model_info(m)['name'] for m in DEFAULT_MODELS],
            "individual_responses": results,
            "consolidated_analysis": consolidation,
            "success_rate": f"{len([r for r in results if r['success']])}/{len(results)}"
        }

@ns.route('/models')
class AvailableModels(Resource):
    def get(self):
        """List available models"""
        return {
            "default_models": DEFAULT_MODELS,
            "popular_models": [
                'mistralai/mistral-7b-instruct',
                'qwen/qwen3-30b-a3b:free',
                'openai/gpt-3.5-turbo',
                'meta-llama/llama-3-8b-instruct'
            ]
        }

if __name__ == '__main__':
    print("Access Swagger UI at: http://localhost:5100/swagger-ui")
    #app.run(host='0.0.0.0', port=5100, debug=True)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))