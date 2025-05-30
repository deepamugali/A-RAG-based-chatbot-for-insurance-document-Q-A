import os
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import re
# Load local model for answer generation
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")

def generate_answer_with_local_model(prompt):
    result = qa_pipeline(prompt, max_new_tokens=256, temperature=0.3)
    return result[0]["generated_text"].strip()

def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local("rag_index", embeddings, allow_dangerous_deserialization=True)



 
def get_answer(query):
    db = load_vector_db()
    retriever = db.as_retriever(search_type="similarity", k=3)
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs])

    # ✅ Log retrieved context
    print("\n=== Retrieved Context ===\n")
    print(context[:1000])  # Print first 1000 chars of the context
    print("\n=== End ===\n")

    if not context or len(context.strip()) < 30:
        return "I don't know"

    # ✅ Check if query shares real keywords with context
    question_keywords = set(re.findall(r'\b\w+\b', query.lower()))
    context_words = set(re.findall(r'\b\w+\b', context.lower()))

    overlap = question_keywords & context_words

    if len(overlap) < 2:  # Require at least 2 shared keywords
        return "I don't know"

    # Build prompt
    prompt = f"""Answer the question based only on the context below.
If the answer isn't in the context, say "I don't know".

Context:
{context}

Question: {query}
Answer:"""

    answer = generate_answer_with_local_model(prompt).strip()

    # ✅ Final safety filter
    if (
        not answer
        or "i don't know" in answer.lower()
        or len(answer.strip()) < 20
        or len(set(re.findall(r'\b\w+\b', answer.lower())) & context_words) < 2
    ):
        return "I don't know"

    return answer




# def get_answer(query):
#     db = load_vector_db()
#     retriever = db.as_retriever(search_type="similarity", k=3)
#     docs = retriever.get_relevant_documents(query)
#     context = "\n".join([doc.page_content for doc in docs])

#     # ✅ Bail out early if nothing relevant was retrieved
#     if not context or len(context.strip()) < 30:
#         return "I don't know"

#     # ✅ Bail out if no overlap between query and context
#     lower_context = context.lower()
#     overlap_words = [word for word in query.lower().split() if word in lower_context]

#     if not overlap_words:
#         return "I don't know"

#     # 🔧 Create prompt for generation
#     prompt = f"""Answer the question based only on the context below.
# If the answer isn't in the context, say "I don't know".

# Context:
# {context}

# Question: {query}
# Answer:"""

#     answer = generate_answer_with_local_model(prompt).strip()

#     # ✅ Block answers that are short, hallucinated, or vague
#     if (
#         not answer
#         or "i don't know" in answer.lower()
#         or len(answer) < 20
#         or not any(word in answer.lower() for word in overlap_words)
#     ):
#         return "I don't know"

#     return answer
