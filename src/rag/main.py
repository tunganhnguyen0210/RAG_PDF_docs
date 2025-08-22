from pydantic import BaseModel, Field

from .file_loader import Loader  # Nhập từ cùng thư mục rag
from .vectorstore import VectorDB
from .offline_rag import OfflineRAG

class InputQA(BaseModel):
    question: str = Field(..., description="The question to ask the RAG system")

class OutputQA(BaseModel):
    answer: str = Field(..., description="The answer to the question")

def build_rag_chain(llm, data_dir, data_type):
    doc_loaded = Loader(file_type=data_type).load_dir(dir_path=data_dir, workers=2)
    retriever = VectorDB(documents=doc_loaded).get_retriever()
    rag_chain = OfflineRAG(llm).get_chain(retriever)

    return rag_chain