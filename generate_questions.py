from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import re

# Load FAISS index
db = FAISS.load_local("rag_index", HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"), allow_dangerous_deserialization=True)

# Extract documents
retriever = db.as_retriever(search_type="similarity", k=20)
docs = retriever.vectorstore.docstore._dict.values()

# Extract meaningful sentences
text_data = "\n".join(doc.page_content for doc in docs)
sentences = list(set(re.findall(r"([A-Z][^\.!?]{20,150}[\.!?])", text_data)))[:10]

# Load question generation pipeline
generator = pipeline("text2text-generation", model="valhalla/t5-small-qg-prepend", tokenizer="t5-small")

questions = []
for sentence in sentences:
    try:
        prompt = f"generate question: {sentence}"
        q = generator(prompt, max_new_tokens=64)[0]['generated_text']
        questions.append(q)
    except Exception as e:
        print("❌ Error:", e)

# Save to a JSON file
import json
with open("suggested_questions.json", "w") as f:
    json.dump(questions, f, indent=2)

print("✅ Suggested questions saved to suggested_questions.json")
