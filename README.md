# VitaGuide — Public Health Promotion & Disease Prevention RAG Chatbot

VitaGuide is an educational Retrieval-Augmented Generation (RAG) chatbot grounded exclusively in publicly available, non-sensitive WHO/PAHO health promotion guidelines. It is built with 100% free and open-source tools (Python, Streamlit, FastEmbed ONNX, Qdrant, and open-weight LLMs via OpenRouter's free tier).

## Local Reproducibility
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OPENROUTER_API_KEY="your_free_openrouter_key"
export QDRANT_URL="your_qdrant_cloud_url"          # Optional (uses local Qdrant if omitted)
export QDRANT_API_KEY="your_qdrant_cloud_api_key"  # Optional
streamlit run app.py
