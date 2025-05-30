


# import os
# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS

# def get_hf_embeddings():
#     return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# def ingest_documents():
#     # Use absolute paths to locate PDFs in the 'docs' folder
#     base_path = os.path.join(os.path.dirname(__file__), "docs")
#     pdf_files = [
#         "America's_Choice_2500_Gold_SOB.pdf",
#         "America's_Choice_5000_Bronze_SOB.pdf",
#         "America's_Choice_5000_HSA_SOB.pdf",
#         "America's_Choice_7350_Copper_SOB.pdf"
#     ]
#     pdf_paths = [os.path.join(base_path, file) for file in pdf_files]

#     # Check that all files exist
#     for path in pdf_paths:
#         loader = PyMuPDFLoader(path)
#         loaded_docs = loader.load()
#         print(f"📄 Loaded {len(loaded_docs)} documents from {path}")
        
#         # Print a sample of the first document's content (if any)
#         if loaded_docs:
#             print("🔍 Sample text:\n", loaded_docs[0].page_content[:300])
    
#     docs.extend(loaded_docs)
#     # Load and split all PDFs into chunks
#     docs = []
#     for path in pdf_paths:
#         loader = PyMuPDFLoader(path)
#         docs.extend(loader.load())

#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     chunks = splitter.split_documents(docs)

#     # Use Hugging Face embeddings instead of OpenAI
#     embeddings = get_hf_embeddings()
#     db = FAISS.from_documents(chunks, embeddings)
#     db.save_local("rag_index")

#     print("✅ PDF ingestion and FAISS vector store created successfully.")

# if __name__ == "__main__":
#     ingest_documents()


import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def get_hf_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def ingest_documents():
    # Base folder containing the PDFs
    base_path = os.path.join(os.path.dirname(__file__), "docs")
    pdf_files = [
        "America's_Choice_2500_Gold_SOB.pdf",
        "America's_Choice_5000_Bronze_SOB.pdf",
        "America's_Choice_5000_HSA_SOB.pdf",
        "America's_Choice_7350_Copper_SOB.pdf"
    ]
    pdf_paths = [os.path.join(base_path, file) for file in pdf_files]

    # Validate files
    for path in pdf_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Missing file: {path}")

    # Load PDFs and extract text
    docs = []
    for path in pdf_paths:
        print(f"📂 Loading: {path}")
        loader = PyMuPDFLoader(path)
        loaded_docs = loader.load()
        print(f"✅ Loaded {len(loaded_docs)} documents from {os.path.basename(path)}")
        
        # Show sample content
        if loaded_docs:
            print("🔍 Sample content:\n", loaded_docs[0].page_content[:300])
        else:
            print("⚠️ No content extracted from this PDF.")
        
        docs.extend(loaded_docs)

    if not docs:
        print("❌ No documents loaded. Aborting.")
        return

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"🧩 Split into {len(chunks)} chunks.")

    if not chunks:
        print("❌ No chunks created. Check if PDF content was empty.")
        return

    # Embed and save FAISS index
    print("🔗 Creating embeddings...")
    embeddings = get_hf_embeddings()
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("rag_index")

    print("✅ PDF ingestion and FAISS vector store created successfully.")

if __name__ == "__main__":
    ingest_documents()
