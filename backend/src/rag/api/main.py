"""FastAPI application — HTTP layer over the RAG pipeline."""
import logging
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import json 
from src.rag.retrieval.retriever import retrieve, RetrievedChunk
from src.rag.synthesis.answer import synthesize_answer
from src.rag.aws.bedrock import HAIKU, SONNET


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
from fastapi.responses import StreamingResponse
from src.rag.agent.graph import agent_graph
from src.rag.aws.bedrock import claude_stream
from src.rag.agent.nodes import SYNTH_SYSTEM






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

@app.post("/ask/stream")
def ask_stream(request: AskRequest):
    """
    Stream the response. Returns SSE.
    
    Flow:
    1. Run the graph up to (but not including) synthesizer
    2. If we hit an apology node, stream the static text
    3. Otherwise, stream synthesis from Claude
    """
    def event_stream():
        # Run pre-synthesis flow synchronously (it's fast)
        state = agent_graph.invoke(
            {"question": request.question},
            config={"configurable": {"skip_synth": True}},
        )
        
        # First: emit metadata (sources) as a JSON event
        sources_payload = json.dumps({
            "type": "sources",
            "sources": [
                {"filename": c["filename"], "page_num": c["page_num"],
                 "score": round(c["rrf_score"], 4)}
                for c in state.get("chunks", [])
            ],
            "trace": state.get("trace", []),
        })
        yield f"event: metadata\ndata: {sources_payload}\n\n"
        
        # Then: stream the answer
        if state.get("answer"):
            # Apology path — just yield the static text
            yield f"event: token\ndata: {json.dumps({'text': state['answer']})}\n\n"
        else:
            # Synthesis path — actually stream from Claude
            chunks = state["chunks"]
            context = "\n---\n".join(
                f"[source_{i+1}] ({c['filename']}, p{c['page_num']})\n{c['content']}"
                for i, c in enumerate(chunks)
            )
            prompt = f"CONTEXT PASSAGES:\n{context}\n\nQUESTION: {request.question}\n\nAnswer:"
            
            for token in claude_stream(prompt, system=SYNTH_SYSTEM):
                yield f"event: token\ndata: {json.dumps({'text': token})}\n\n"
        
        yield "event: done\ndata: {}\n\n"
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")
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