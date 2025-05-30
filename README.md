# Insurance RAG Chatbot (Flask-based)

This is a Python Flask application that implements a Retrieval-Augmented Generation (RAG) chatbot. The chatbot answers insurance-related questions strictly based on the provided PDF documents.

## 📦 Project Structure

- `app.py`: Flask backend API
- `rag_bot.py`: Logic for loading the vector index and answering questions
- `pdf_ingest.py`: Script for ingesting PDFs and building the vector store
- `requirements.txt`: Python dependencies

## ⚙️ Setup Instructions

### 1. Clone the repo and install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your OpenAI API key
Set it as an environment variable:
```bash
export OPENAI_API_KEY='your-api-key'
```

### 3. Place your PDFs in the `docs/` folder and run:
```bash
python pdf_ingest.py
```

### 4. Start the Flask app:
```bash
python app.py
```

### 5. Ask a question via cURL or Postman:
```bash
curl -X POST http://127.0.0.1:5000/ask -H "Content-Type: application/json" -d '{"question": "What is the deductible for the 2500 Gold plan?"}'
```

## 🚫 Out-of-Scope Questions

If a question is not covered in the PDFs, the bot will respond with:

```json
"I don't know"
```
