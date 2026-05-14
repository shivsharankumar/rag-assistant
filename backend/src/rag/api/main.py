"""FastAPI application — HTTP layer over the RAG pipeline."""
import logging
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.rag.retrieval.retriever import retrieve, RetrievedChunk
from src.rag.synthesis.answer import synthesize_answer
from src.rag.aws.bedrock import HAIKU, SONNET


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


app = FastAPI(
    title="RAG Assistant API",
    description="Ask questions about research papers.",
    version="0.2.0",
)

# Allow local dev frontend (Week 5)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Request/response schemas ----

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    model: Literal["haiku", "sonnet"] = "haiku"


class Source(BaseModel):
    chunk_id: int
    filename: str
    page_num: int
    score: float
    preview: str  # first 200 chars


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    model_used: str


# ---- Routes ----

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    log.info(f"Question: {request.question!r}  top_k={request.top_k}  model={request.model}")
    
    try:
        chunks = retrieve(request.question, top_k=request.top_k)
    except Exception as e:
        log.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")
    
    if not chunks:
        return AskResponse(
            answer="No relevant content found in the knowledge base.",
            sources=[],
            model_used=request.model,
        )
    
    model_id = HAIKU if request.model == "haiku" else SONNET
    
    try:
        answer = synthesize_answer(request.question, chunks, model_id=model_id)
    except Exception as e:
        log.exception("Synthesis failed")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {e}")
    
    sources = [
        Source(
            chunk_id=c["chunk_id"],
            filename=c["filename"],
            page_num=c["page_num"],
            score=c["score"],
            preview=c["content"][:200],
        )
        for c in chunks
    ]
    
    return AskResponse(answer=answer, sources=sources, model_used=request.model)