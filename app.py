import os
import warnings
import logging
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

# --- CẤU HÌNH DỌN DẸP LOG ---
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)

# 1. Cấu hình giao diện trang
st.set_page_config(page_title="RAG Chatbot Doanh Nghiệp", page_icon="🤖", layout="wide")

load_dotenv()

# --- HÀM XỬ LÝ DỮ LIỆU ---
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

def process_documents(uploaded_files):
    all_chunks = []
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        chunks = text_splitter.split_documents(documents)
        all_chunks.extend(chunks)
        
    # --- THÊM CHỐT CHẶN AN TOÀN TẠI ĐÂY ---
    if not all_chunks:
        raise ValueError("Không đọc được chữ nào trong file PDF! Vui lòng đảm bảo file không phải là ảnh scan hoặc bị khóa mã hóa.")
    # --------------------------------------
        
    embeddings = load_embedding_model()
    vector_store = FAISS.from_documents(all_chunks, embeddings)
    return vector_store, all_chunks

# 🌟 PHASE 5 UPDATE: Hàm sinh câu hỏi gợi ý tự động
def generate_suggested_questions(chunks):
    # 1. Trộn data: Lấy 2 đoạn ở đầu và 3 đoạn ở giữa tài liệu để có cái nhìn bao quát hơn
    mid_index = len(chunks) // 2
    sample_chunks = chunks[:2] + chunks[mid_index : mid_index + 3]
    
    # Ép thành chuỗi text, giới hạn 2500 ký tự để không quá tải
    sample_text = " ".join([c.page_content for c in sample_chunks])[:2500] 
    
    # 2. SỬA LẠI TÊN MODEL chuẩn của Google
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.7)
    
    prompt = f"""Dựa vào nội dung sau đây được trích xuất từ một tài liệu, hãy gợi ý đúng 3 câu hỏi ngắn gọn (dưới 15 chữ) bằng tiếng Việt mà người dùng có thể hỏi để tìm hiểu sâu hơn về tài liệu này. 
    Chỉ trả về 3 câu hỏi, mỗi câu bắt đầu bằng dấu gạch ngang (-). Tuyệt đối không giải thích thêm.
    
    Nội dung tài liệu: 
    {sample_text}"""
    
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        # 3. In lỗi ra terminal để bạn biết đường mà sửa nếu API có vấn đề
        print(f"❌ Lỗi khi sinh câu hỏi gợi ý: {e}") 
        return "- Nội dung chính của tài liệu là gì?\n- Có điểm nào quan trọng cần lưu ý?\n- Tóm tắt lại tài liệu này."

def build_qa_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
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
    
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "chat_history", "question"])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.2)
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory, combine_docs_chain_kwargs={"prompt": PROMPT}, return_source_documents=True
    )
    return qa_chain

# --- GIAO DIỆN STREAMLIT ---
st.title("🤖 Chatbot RAG - Trợ lý Tài liệu")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "suggestions" not in st.session_state:
    st.session_state.suggestions = ""

with st.sidebar:
    st.header("📂 Quản lý tài liệu")
    # 🌟 PHASE 5 UPDATE: Thêm tham số accept_multiple_files=True
    uploaded_files = st.file_uploader("Tải lên tài liệu PDF (Có thể chọn nhiều file)", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Xử lý dữ liệu"):
            with st.spinner("Đang đọc, băm nhỏ và tạo trí nhớ cho tài liệu..."):
                vector_store, all_chunks = process_documents(uploaded_files)
                st.session_state.qa_chain = build_qa_chain(vector_store)
                
                # Sinh câu hỏi gợi ý sau khi xử lý xong
                st.session_state.suggestions = generate_suggested_questions(all_chunks)
                st.success(f"Đã xử lý xong {len(uploaded_files)} tài liệu!")
                
    st.markdown("---")
    if st.button("🗑️ Xóa lịch sử trò chuyện"):
        st.session_state.messages = []
        st.session_state.suggestions = ""
        if st.session_state.qa_chain:
            st.session_state.qa_chain.memory.clear()
        st.rerun()

# 🌟 PHASE 5 UPDATE: Hiển thị câu hỏi gợi ý
if st.session_state.qa_chain and st.session_state.suggestions and len(st.session_state.messages) == 0:
    raw_suggestions = st.session_state.suggestions
    display_text = ""
    
    # Kỹ thuật bóc tách phần lõi văn bản
    if isinstance(raw_suggestions, str):
        display_text = raw_suggestions
    elif isinstance(raw_suggestions, list):
        for item in raw_suggestions:
            if isinstance(item, dict) and "text" in item:
                display_text += item["text"] + "\n"
            else:
                display_text += str(item) + "\n"
                
    st.info("💡 **Gợi ý câu hỏi cho bạn:**\n" + display_text)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Hãy hỏi nội dung trong tài liệu PDF..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
        
    if not st.session_state.qa_chain:
        with st.chat_message("assistant"):
            st.warning("⚠️ Vui lòng tải lên và ấn 'Xử lý dữ liệu' file PDF ở menu bên trái trước khi chat nhé!")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Đang lục lọi tài liệu..."):
                try:
                    response = st.session_state.qa_chain.invoke({"question": query})
                    answer = response["answer"]
                    sources = response["source_documents"]
                    
                    st.markdown(answer)
                    
                    with st.expander("🔍 Xem nguồn trích xuất từ PDF"):
                        for i, doc in enumerate(sources):
                            st.info(f"**Đoạn {i+1} (Từ file: {os.path.basename(doc.metadata.get('source', 'Không rõ'))}):** {doc.page_content[:250]}...")
                            
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Đã xảy ra lỗi: {e}")