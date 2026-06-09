import os
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # Sửa import
from langchain_community.vectorstores import FAISS

# Đảm bảo stdout ghi dưới dạng utf-8 trên Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

def ingest_document(pdf_path, vector_db_path="faiss_index"):
    print(f"[INFO] 1. Load tai lieu tu: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print("[INFO] 2. Chia nho tai lieu (Chunking) voi cau hinh toi uu (1500/300)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   -> Da chia thanh {len(chunks)} chunks.")
    
    print("[INFO] 3. Tao Embeddings bang HuggingFace va luu vao FAISS...")
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(vector_db_path)
    
    print(f"[INFO] 4. Hoan tat! FAISS index da duoc luu tai thu muc: ./{vector_db_path}/")

if __name__ == "__main__":
    FILE_PDF = "data/RungNauy.pdf" 
    
    if os.path.exists(FILE_PDF):
        ingest_document(FILE_PDF)
    else:
        print(f"[ERROR] Khong tim thay file {FILE_PDF}.")