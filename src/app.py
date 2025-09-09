import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from langserve import add_routes

from src.base.llm import get_gemini_llm
from src.rag.main import build_rag_chain, InputQA, OutputQA

llm = get_gemini_llm()
agent_docs = "./RAG_langchain/data_src/AI_Agent"


# ----------Chains-----------
rag_chain = build_rag_chain(llm, data_dir=agent_docs, data_type="pdf")

# ----------FastAPI-----------

app = FastAPI(
    title="RAG Langchain Server",
    version="1.0",
    description="A server for RAG Langchain with Gemini LLM",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],    
    expose_headers=["*"],
)

# -----Routes - FastAPI -----

@app.get("/check")
async def check():
    return {"status": "ok"}

@app.post("/rag_chain", response_model=OutputQA)
async def rag_ai(input: InputQA):
    answer = rag_chain.invoke(input.question)
    return { "answer": answer }


# ----Langserve Routes - Playground------
add_routes(
    app,
    rag_chain,
    playground_type="default",
    path="/rag_agent_ai"
)
