import os
import sys
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

# Đảm bảo stdout ghi dưới dạng utf-8 trên Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# 1. Load API Key từ file .env
load_dotenv()

# 2. Khởi tạo lại Embedding Model (Phải GIỐNG HỆT model đã dùng ở Phase 2)
# Nếu ở Phase 2 dùng Gemini, hãy thay bằng GoogleGenerativeAIEmbeddings(...)
print("[INFO] Dang load mo hinh Embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

# Load FAISS index từ ổ cứng
print("[INFO] Dang load FAISS Vector Database...")
vector_store = FAISS.load_local(
    "faiss_index", 
    embeddings, 
    allow_dangerous_deserialization=True # Cần thiết cho các bản LangChain mới
)

# Tạo Retriever: Cấu hình chỉ lấy top 3 chunks (đoạn text) liên quan nhất
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 3. Viết Custom Prompt Template 
prompt_template = """
Bạn là một trợ lý AI đọc hiểu tài liệu chuyên nghiệp. 
Hãy sử dụng các thông tin được cung cấp trong phần "Tài liệu tham khảo" dưới đây để trả lời câu hỏi của người dùng.
Luật tối thượng: NẾU BẠN KHÔNG TÌM THẤY CÂU TRẢ LỜI TRONG TÀI LIỆU, HÃY TRẢ LỜI CHÍNH XÁC LÀ: "Tôi không tìm thấy thông tin này trong tài liệu", tuyệt đối không được tự bịa ra thông tin.

Tài liệu tham khảo (Context):
{context}

Lịch sử trò chuyện:
{chat_history}

Câu hỏi của người dùng: {question}
Câu trả lời của bạn:"""

PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "chat_history", "question"]
)

# 4. Thêm ConversationBufferMemory để nhớ lịch sử chat
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# 5. Khởi tạo LLM (Dùng Gemini 1.5 Flash)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite", 
    temperature=0.2 # Temperature thấp để câu trả lời bám sát sự thật, bớt "bay bổng"
)

# 6. Lắp ráp RAG Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": PROMPT},
    return_source_documents=True # Trả về cả các chunk đã dùng để kiểm tra chéo
)

# 7. Hàm chạy giao diện dòng lệnh (CLI)
def chat_cli():
    print("\n" + "="*50)
    print("[INFO] Bot: RAG Chatbot da san sang! (Go 'exit' hoac 'quit' de thoat)")
    print("="*50 + "\n")
    
    while True:
        try:
            query = input("Ban: ")
        except UnicodeDecodeError:
            # Phòng trường hợp đầu vào bàn phím bị lỗi mã hóa
            query = ""
            
        if query.lower() in ['exit', 'quit']:
            print("Bot: Tam biet nhe!")
            break
            
        if not query.strip():
            continue
            
        # Đẩy câu hỏi vào Chain
        try:
            result = qa_chain.invoke({"question": query})
            print(f"Bot: {result['answer']}")
            
            print("\n[Nguon trich xuat]:")
            for i, doc in enumerate(result['source_documents']):
                # In ra an toàn
                content_preview = doc.page_content[:100].replace('\n', ' ')
                print(f"  - Chunk {i+1}: {content_preview}...")
                
        except Exception as e:
            print(f"[ERROR] Loi: {e}")
            
        print("-" * 50)

if __name__ == "__main__":
    chat_cli()