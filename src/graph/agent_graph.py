from langgraph.graph import END, StateGraph

from ..nodes.abstain_node import abstain_node
from ..nodes.generate_node import generate_node
from ..nodes.rerank_node import rerank_node
from ..nodes.retrieve_node import retrieve_node
from .state import RAGState


def route_after_abstain(state: RAGState) -> str:
    # return END if state["abstain"] else "generate"
    if state.get("abstain"):
        return END
    else:
        return "generate"


graph = StateGraph(RAGState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("rerank", rerank_node)
graph.add_node("abstain", abstain_node)
graph.add_node("generate", generate_node)

graph.set_entry_point("retrieve")

graph.add_edge("retrieve", "rerank")
graph.add_edge("rerank", "abstain")

graph.add_conditional_edges("abstain", route_after_abstain, {"generate": "generate", END: END})

graph.add_edge("generate", END)

app = graph.compile()

try:
    app.get_graph().draw_mermaid_png(output_file_path="graph.png")
except Exception:
    pass
