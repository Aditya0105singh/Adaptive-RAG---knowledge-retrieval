"""LangGraph StateGraph wiring for the adaptive RAG pipeline."""
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents.edges import decide_after_grading, decide_route
from src.agents.nodes import (
    generate_answer,
    grade_documents,
    retrieve_docs,
    route_question,
    transform_query,
    web_search_node,
)
from src.agents.state import GraphState


def build_graph():
    """Construct and compile the adaptive RAG StateGraph with a MemorySaver checkpointer."""
    workflow = StateGraph(GraphState)

    workflow.add_node("route_question", route_question)
    workflow.add_node("retrieve", retrieve_docs)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    # Node name must differ from the 'web_search' state key (LangGraph constraint).
    workflow.add_node("do_web_search", web_search_node)
    workflow.add_node("generate", generate_answer)

    workflow.add_edge(START, "route_question")
    workflow.add_conditional_edges(
        "route_question",
        decide_route,
        {
            "index": "retrieve",
            "general": "generate",
            "search": "do_web_search",
        },
    )
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {
            "generate": "generate",
            "transform_query": "transform_query",
            "web_search": "do_web_search",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("do_web_search", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile(checkpointer=MemorySaver())


app = build_graph()
