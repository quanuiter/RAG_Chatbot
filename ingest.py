import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # Sửa import
from langchain_community.vectorstores import FAISS

def ingest_document(pdf_path, vector_db_path="faiss_index"):
    print(f"1. Đang load tài liệu từ: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print("2. Đang chia nhỏ tài liệu (Chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   -> Đã chia thành {len(chunks)} chunks.")
    
    print("3. Đang tạo Embeddings bằng HuggingFace (Local) và lưu vào FAISS...")
    # Dùng model mã nguồn mở, nhẹ và tối ưu cho CPU
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(vector_db_path)
    
    print(f"4. Hoàn tất! FAISS index đã được lưu tại thư mục: ./{vector_db_path}/")

if __name__ == "__main__":
    FILE_PDF = "data/RungNauy.pdf" 
    
    if os.path.exists(FILE_PDF):
        ingest_document(FILE_PDF)
    else:
        print(f"Lỗi: Không tìm thấy file {FILE_PDF}.")