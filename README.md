# Tối ưu hóa LLM cho Hệ thống Hỏi-Đáp Tài liệu Doanh nghiệp (RAG Chatbot)

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1%2B-green.svg)](https://www.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Tổng quan

Repository này chứa mã nguồn của một hệ thống Hỏi-Đáp Tài liệu Doanh nghiệp (Enterprise Document QA Chatbot) được tối ưu hóa, xây dựng trên kiến trúc Retrieval-Augmented Generation (RAG).

Dự án giải quyết bài toán tra cứu thông tin tốn thời gian trong các văn bản nội bộ—nội quy, chính sách, hợp đồng—bằng cách chuyển đổi toàn bộ tài liệu PDF thành cơ sở dữ liệu vector và sử dụng Large Language Models (LLM) để tổng hợp câu trả lời chính xác, có cơ sở. Nguyên tắc thiết kế cốt lõi của hệ thống là triệt tiêu hoàn toàn hiện tượng ảo giác (Hallucination) của AI: mọi câu trả lời đều được neo vào một đoạn nguồn có thật và có thể truy xuất, đảm bảo tính minh bạch tuyệt đối trong toàn bộ quá trình truy hồi.

---

## Các đóng góp kỹ thuật cốt lõi

* **Tối ưu hóa chi phí và độ trễ bằng Local Embeddings:** Thay vì phụ thuộc vào các API nhúng trả phí (ví dụ: OpenAI Embeddings) với giới hạn rate limit và chi phí per-token, pipeline xử lý tài liệu chạy mô hình `paraphrase-multilingual-MiniLM-L12-v2` từ HuggingFace hoàn toàn trên phần cứng cục bộ. Mô hình này tạo ra các vector 384 chiều với độ phân giải ngữ nghĩa cao cho văn bản tiếng Việt, cho phép xử lý khối lượng lớn tài liệu mà không phụ thuộc bất kỳ API bên ngoài nào.

* **Pipeline xử lý tài liệu đầu vào mạnh mẽ:** `PyPDFLoader` kết hợp với `RecursiveCharacterTextSplitter` được cấu hình ở `chunk_size=1000` và `chunk_overlap=200`. Cửa sổ overlap được chọn kích thước chính xác để bảo toàn tính liên kết ngữ nghĩa giữa các điều khoản trong tài liệu pháp lý và chính sách dài, ngăn chặn sự đứt gãy ngữ cảnh tại ranh giới các chunk.

* **Prompt Engineering chống ảo giác (Anti-Hallucination):** Một system prompt tùy chỉnh nghiêm ngặt được tiêm vào Combine Docs chain. LLM bị ràng buộc bởi một điều kiện cứng: khi ngữ cảnh được truy hồi không chứa đủ thông tin, mô hình buộc phải phản hồi đúng một cụm từ duy nhất là *"Tôi không tìm thấy thông tin này trong tài liệu"* mà không thêm bất kỳ nội dung nào khác. Ràng buộc này được củng cố bằng cách đặt `temperature = 0.2` để triệt tiêu tính sáng tạo tự do, ưu tiên tính chính xác thực tế.

* **Bộ nhớ hội thoại có trạng thái (Stateful Conversational Memory):** `ConversationBufferMemory` của LangChain duy trì trạng thái `chat_history` xuyên suốt các lượt trao đổi, cho phép mô hình giải quyết đúng các tham chiếu đại từ và câu hỏi tiếp nối trong hội thoại đa lượt (ví dụ: *"Vậy điều khoản đó áp dụng cho ai?"*) mà không mất ngữ cảnh trước đó.

* **Giao diện nâng cao với tính năng Onboarding hỗ trợ bởi AI:** Giao diện Streamlit hỗ trợ xử lý đa tài liệu đồng thời. Ngay khi tải file xong, hệ thống tự động lấy mẫu 2.500 ký tự phân bổ đều trên toàn bộ tài liệu và yêu cầu LLM sinh ra 3 câu hỏi gợi ý phù hợp với ngữ cảnh, giúp người dùng không chuyên bắt đầu phiên làm việc hiệu quả ngay lập tức. Tính năng Source Highlighting trình bày rõ ràng đoạn văn nguồn chính xác mà AI đã dùng để tổng hợp mỗi câu trả lời.

---

## Kiến trúc & Tham số hệ thống

### Mô-đun Truy hồi Vector

| Thành phần | Cấu hình |
| :--- | :--- |
| Cơ sở dữ liệu Vector | FAISS (`IndexFlatL2`) |
| Mô hình Nhúng (Embedding) | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Số chiều Vector | 384 |
| Số lượng kết quả truy hồi | `k=3` (truy vấn thông thường) / `k=6` (truy vấn tóm tắt diện rộng) |
| Chiến lược phân đoạn | `RecursiveCharacterTextSplitter`, size=1000, overlap=200 |

### Mô-đun Sinh câu trả lời

| Thành phần | Cấu hình |
| :--- | :--- |
| Large Language Model | Google Gemini 3.5 Flash (`models/gemini-3.5-flash`) |
| Nhiệt độ (Temperature) | 0.2 |
| Bộ nhớ hội thoại | `ConversationBufferMemory` (LangChain) |
| Loại Chain | `ConversationalRetrievalChain` với Combine Docs prompt tùy chỉnh |

Gemini 3.5 Flash được lựa chọn nhờ độ trễ suy luận thấp và khả năng đọc hiểu tốt trên văn bản chính sách tiếng Việt. (đặc biệt hơn là nó miễn phí)

---

## Cấu trúc Repository

```text
RAG_Chatbot/
├── app.py                    # Giao diện Streamlit chính và bộ điều phối toàn bộ RAG pipeline
├── qa_chain.py               # Script CLI độc lập để kiểm thử logic RAG cốt lõi
├── ingest.py                 # Script thử nghiệm phân đoạn tài liệu và lưu FAISS Index
├── requirements.txt          # Danh sách thư viện Python phụ thuộc
├── .gitignore                # Loại trừ môi trường ảo và file API key
└── README.md
```

---

## Hướng dẫn cài đặt & Khởi chạy

### 1. Yêu cầu hệ thống

Clone repository và cài đặt toàn bộ thư viện phụ thuộc vào môi trường ảo riêng biệt:

```bash
git clone https://github.com/quanuiter/RAG_Chatbot.git
cd RAG_Chatbot

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Các thư viện cốt lõi bao gồm `torch` và `transformers`, cần thiết để chạy mô hình nhúng câu cục bộ mà không cần gọi API từ xa.

### 2. Cấu hình biến môi trường

Tạo file `.env` tại thư mục gốc của dự án và điền Google Gemini API key:

```
GOOGLE_API_KEY="api_key_cua_ban"
```

### 3. Khởi chạy giao diện

```bash
streamlit run app.py
```

Ứng dụng sẽ chạy tại địa chỉ `http://localhost:8501/`. Quy trình sử dụng gồm 3 bước:

1. Kéo thả một hoặc nhiều file PDF vào thanh bên trái.
2. Nhấn **"Xử lý dữ liệu"** và chờ pipeline vector hóa hoàn tất.
3. Sử dụng các câu hỏi gợi ý được sinh tự động hoặc nhập câu hỏi tùy chỉnh.

---

## Triển khai

Dự án được tối ưu hóa để triển khai dạng serverless trên **Streamlit Community Cloud**. `GOOGLE_API_KEY` và các thông tin bí mật khác được quản lý an toàn qua tính năng **Advanced Settings > Secrets** của nền tảng, đảm bảo không có thông tin nhạy cảm nào được đẩy lên repository.

Live Demo: *[Đường link Streamlit Cloud]*

---

## Tài liệu tham khảo

* **Mô hình Nhúng:** [`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) — Reimers & Gurevych, *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*, EMNLP 2019.
* **Tìm kiếm Vector:** [FAISS](https://github.com/facebookresearch/faiss) — Johnson et al., *Billion-scale similarity search with GPUs*, IEEE Transactions on Big Data, 2019.
* **Framework điều phối:** [LangChain](https://www.langchain.com/) cho việc tổ hợp chain và quản lý bộ nhớ hội thoại.
* **Mô hình ngôn ngữ:** [Google Gemini 1.5 Flash](https://deepmind.google/technologies/gemini/) thông qua Google Generative AI SDK.

---

*Phát triển bởi Ngô Nhật Quân — Trường Đại học Công nghệ Thông tin, ĐHQG TP.HCM*
