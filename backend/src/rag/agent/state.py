"""Shared state object that flows through the LangGraph."""
from typing import TypedDict, Literal
from src.rag.retrieval.hybrid import RetrievedChunk


QuestionType = Literal["factual", "comparison", "summary", "out_of_scope"]


class AgentState(TypedDict, total=False):
    # Input
    question: str
    
    # Set by router
    question_type: QuestionType
    
    # Set by query rewriter
    rewritten_question: str
    
    # Set by retriever
    chunks: list[RetrievedChunk]
    
    # Set by grounding check
    is_grounded: bool
    grounding_reason: str
    
    # Set by synthesizer
    answer: str
    
    # For tracing / debugging
    trace: list[str]  # log of which nodes executed
    use_critic: bool
    critic_score: float          # 0-10
    critic_reasoning: str        # explanation
    critic_unsupported_claims: list[str]