# 📄 RAG PDF Documents - Question Answering System

A **Retrieval-Augmented Generation (RAG)** system that enables intelligent question answering over PDF documents using **Google Gemini**, **LangChain**, and **FastAPI**.

---

## ✨ Features

- 🔍 **PDF Document Ingestion** — Load and parse PDF files (including image extraction) with parallel multiprocessing
- 🧠 **Vector Store Retrieval** — Chunk documents and store embeddings using **Chroma** or **FAISS** with HuggingFace embeddings
- 🤖 **Gemini LLM Integration** — Powered by **Google Gemini 2.0 Flash** for fast, accurate responses
- 🚀 **FastAPI Server** — RESTful API endpoint for querying the RAG pipeline
- 🎮 **LangServe Playground** — Built-in interactive playground for testing queries

---

## 📁 Project Structure

```
RAG_PDF_docs/
├── .env                        # Environment variables (API keys)
├── requirements.txt            # Python dependencies
├── data_src/
│   └── AI_Agent/               # PDF documents for knowledge base
│       ├── 2506.23978v2.pdf
│       ├── 2507.01701v1.pdf
│       ├── 2507.05723v1.pdf
│       ├── 2507.11810v1.pdf
│       └── 2507.11988v2.pdf
└── src/
    ├── app.py                  # FastAPI application entry point
    ├── base/
    │   └── llm.py              # Gemini LLM configuration
    └── rag/
        ├── main.py             # RAG chain builder & Pydantic schemas
        ├── file_loader.py      # PDF loader with multiprocessing & text splitter
        ├── offline_rag.py      # RAG chain (prompt + LLM + output parser)
        ├── vectorstore.py      # Vector database (Chroma/FAISS)
        └── utils.py            # Utility functions
```

---

## 🛠️ Tech Stack

| Component       | Technology                          |
| --------------- | ----------------------------------- |
| **LLM**         | Google Gemini 2.0 Flash             |
| **Framework**   | LangChain                          |
| **Embeddings**  | HuggingFace Embeddings              |
| **Vector Store**| Chroma (default) / FAISS            |
| **API Server**  | FastAPI + LangServe                 |
| **PDF Parsing** | PyPDFLoader (with image extraction) |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/tunganhnguyen0210/RAG_PDF_docs.git
cd RAG_PDF_docs
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root (or edit the existing one):

```env
GOOGLE_API_KEY="your_google_api_key_here"
```

> 💡 Get your API key from [Google AI Studio](https://aistudio.google.com/apikey)

### 4. Add Your PDF Documents

Place your PDF files in the `data_src/AI_Agent/` directory.

### 5. Run the Server

```bash
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

---

## 📡 API Endpoints

### Health Check

```http
GET /check
```

**Response:**
```json
{ "status": "ok" }
```

### Ask a Question

```http
POST /rag_chain
Content-Type: application/json

{
  "question": "What is an AI Agent?"
}
```

**Response:**
```json
{
  "answer": "An AI Agent is..."
}
```

### LangServe Playground

Visit [http://localhost:8000/rag_agent_ai/playground](http://localhost:8000/rag_agent_ai/playground) for an interactive testing UI.

---

## ⚙️ Configuration

### LLM Settings

Edit `src/base/llm.py` to customize the Gemini model:

| Parameter            | Default            | Description              |
| -------------------- | ------------------ | ------------------------ |
| `model_name`         | `gemini-2.0-flash` | Gemini model variant     |
| `temperature`        | `0.3`              | Response creativity      |
| `max_output_tokens`  | `1024`             | Max response length      |

### Document Chunking

Edit `src/rag/file_loader.py` to adjust text splitting:

| Parameter       | Default | Description                        |
| --------------- | ------- | ---------------------------------- |
| `chunk_size`    | `300`   | Number of characters per chunk     |
| `chunk_overlap` | `0`     | Overlap between consecutive chunks |

### Retriever Settings

Edit `src/rag/vectorstore.py` to adjust retrieval:

| Parameter     | Default        | Description                       |
| ------------- | -------------- | --------------------------------- |
| `search_type` | `"similarity"` | Search algorithm                  |
| `k`           | `10`           | Number of relevant chunks to retrieve |

---

## 🔄 Pipeline Architecture

The RAG pipeline follows a streamlined, 5-step process:

1. **📄 Document Loading**: `PyPDFLoader` reads PDF files from `data_src/` (including extracting images and cleaning text).
2. **✂️ Text Splitting**: `RecursiveCharacterTextSplitter` breaks the documents into smaller, semantic chunks (default 300 characters).
3. **🗄️ Vector Storage**: Chunks are embedded using HuggingFace and stored in a Vector DB (`Chroma` or `FAISS`).
4. **🔍 Retrieval**: When a question is asked, the system retrieves the top-k most relevant chunks from the database.
5. **🤖 Generation**: The Gemini 2.0 Flash LLM synthesizes the retrieved context and question to generate a precise answer.

---

## 📝 License

This project is for educational and research purposes.

---

## 👤 Author

**Tung Anh Nguyen** — [@tunganhnguyen0210](https://github.com/tunganhnguyen0210)
