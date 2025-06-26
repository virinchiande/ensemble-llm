# Calcuttapuri LLM Ensemble Web App

A clean Flask web application for querying multiple LLMs and getting consolidated responses.

## 🚀 Quick Start

1. **Extract the project files**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set your OpenRouter API key:**
   ```bash
   export OPENROUTER_API_KEY="your_api_key_here"
   ```
   (Windows: `set OPENROUTER_API_KEY="your_api_key_here"`)

4. **Run the app:**
   ```bash
   python app.py
   ```

5. **Open your browser:**
   - Main website: http://localhost:5000/
   - LLM Ensemble: http://localhost:5000/llm_ensemble

## 📁 Project Structure

```
calcuttapuri-llm-ensemble/
├── app.py                    # Main Flask application
├── templates/
│   ├── index.html           # Main website with navigation
│   └── llm_ensemble.html    # LLM form and results page
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## ✨ Features

- **Beautiful responsive design** with sidebar navigation
- **LLM Ensemble page** with form submission
- **Real-time results** displayed on the same page
- **Multiple model support** (DeepSeek, Llama, etc.)
- **Consolidated analysis** of all responses
- **Mobile-friendly** design
- **Error handling** for failed API calls

## 🔧 Configuration

The app uses these LLM models by default:
- DeepSeek Chat v3
- Meta Llama 3.3 70B

You can modify the `DEFAULT_MODELS` list in `app.py` to use different models.

## 🌐 Usage

1. Go to the main page to see the portfolio/navigation
2. Click "LLM Ensemble" or visit `/llm_ensemble`
3. Enter your question in the text area
4. Click "Query LLMs" to get responses from multiple models
5. View individual responses and consolidated analysis

No API documentation or Swagger UI - just a clean web interface!
