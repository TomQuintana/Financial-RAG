from .agent_graph import app as agent_graph
from .state import RAGState


class GraphService:
    """Runs the LangGraph RAG pipeline and returns a structured response."""

    def __init__(self):
        self.graph = agent_graph

    def process_query(self, user_message: str, metadata: dict | None = None) -> dict:
        try:
            initial_state: RAGState = {
                "query": user_message,
                "documents": [],
                "scores": [],
                "answer": "",
                "abstain": False,
            }
            result = self.graph.invoke(initial_state)

            return {
                "response": (
                    "No encontré información relevante en los reportes financieros."
                    if result.get("abstain")
                    else result.get("answer") or "No se generó respuesta."
                ),
                "success": True,
                "error": None,
                "metadata": metadata or {},
            }

        except Exception as e:
            return {
                "response": "Error inesperado al procesar la consulta.",
                "success": False,
                "error": str(e),
                "metadata": metadata or {},
            }

    def get_graph_visualization(self) -> str:
        try:
            return self.graph.get_graph().draw_mermaid()
        except Exception as e:
            return f"Error generando visualización: {e}"


graph_service = GraphService()
