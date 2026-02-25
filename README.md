# 🪖 VVA Veteran Archive Explorer (1999)
A RAG-based search tool for the 1999 VVA Archives, powered by **Gemma 3** and LangChain.

## 📸 Demo
<img src="demo.png" alt="Demo">

## 🚀 Setup & Running
1. **Activate Environment:** `source .venv/bin/activate`
2. **Run the App:** `streamlit run app.py`

## ⚙️ Hardware Specs
* **CPU:** Intel i7-10700
* **RAM:** 121GB (Optimized for Gemma 3 27B)

## 🤖 Required Models
Ensure you have pulled these via Ollama:
* `ollama pull gemma3:1b`
* `ollama pull gemma3:4b`
* `ollama pull gemma3:12b`
* `ollama pull gemma3:27b`
* `ollama pull nomic-embed-text`