"""LangGraph composition: nodes + edges + compiled graph."""
from langgraph.graph import StateGraph, END
from src.rag.agent.state import AgentState
from src.rag.agent.nodes import (
    router, query_rewriter, retriever, reranker_node,
    grounding_check, synthesizer, critic,  # ← add
    apologize_out_of_scope, apologize_ungrounded,
)


def build_graph():
    g = StateGraph(AgentState)
    
    g.add_node("router", router)
    g.add_node("query_rewriter", query_rewriter)
    g.add_node("retriever", retriever)
    g.add_node("reranker", reranker_node)  # ← add
    g.add_node("grounding_check", grounding_check)
    g.add_node("synthesizer", synthesizer)
    g.add_node("apologize_out_of_scope", apologize_out_of_scope)
    g.add_node("apologize_ungrounded", apologize_ungrounded)
    g.add_node("critic", critic)
    g.set_entry_point("router")
    
    g.add_conditional_edges("router", route_after_router, {
        "apologize_out_of_scope": "apologize_out_of_scope",
        "query_rewriter": "query_rewriter",
    })
    g.add_edge("query_rewriter", "retriever")
    g.add_edge("retriever", "reranker")          # ← changed
    g.add_edge("reranker", "grounding_check")    # ← new
    g.add_conditional_edges("grounding_check", route_after_grounding, {
        "synthesizer": "synthesizer",
        "apologize_ungrounded": "apologize_ungrounded",
    })
    g.add_conditional_edges("synthesizer", route_after_synth, {
        "critic": "critic",
        END: END,
    })
    g.add_edge("critic", END)           
    g.add_edge("apologize_out_of_scope", END)
    g.add_edge("apologize_ungrounded", END)
    
    return g.compile()

def route_after_synth(state: AgentState) -> str:
    return "critic" if state.get("use_critic") else END
def route_after_router(state: AgentState) -> str:
    """Conditional edge: where to go after classification."""
    if state["question_type"] == "out_of_scope":
        return "apologize_out_of_scope"
    return "query_rewriter"


def route_after_grounding(state: AgentState) -> str:
    """Conditional edge: where to go after grounding check."""
    if state.get("is_grounded"):
        return "synthesizer"
    return "apologize_ungrounded"


# def build_graph():
#     g = StateGraph(AgentState)
    
#     # Nodes
#     g.add_node("router", router)
#     g.add_node("query_rewriter", query_rewriter)
#     g.add_node("retriever", retriever)
#     g.add_node("grounding_check", grounding_check)
#     g.add_node("synthesizer", synthesizer)
#     g.add_node("apologize_out_of_scope", apologize_out_of_scope)
#     g.add_node("apologize_ungrounded", apologize_ungrounded)
    
#     # Entry
#     g.set_entry_point("router")
    
#     # Edges
#     g.add_conditional_edges(
#         "router",
#         route_after_router,
#         {
#             "apologize_out_of_scope": "apologize_out_of_scope",
#             "query_rewriter": "query_rewriter",
#         },
#     )
#     g.add_edge("query_rewriter", "retriever")
#     g.add_edge("retriever", "grounding_check")
#     g.add_conditional_edges(
#         "grounding_check",
#         route_after_grounding,
#         {
#             "synthesizer": "synthesizer",
#             "apologize_ungrounded": "apologize_ungrounded",
#         },
#     )
#     g.add_edge("synthesizer", END)
#     g.add_edge("apologize_out_of_scope", END)
#     g.add_edge("apologize_ungrounded", END)
    
#     return g.compile()


# Module-level singleton — compile once on import
agent_graph = build_graph()
