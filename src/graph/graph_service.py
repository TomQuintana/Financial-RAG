"""Service wrapper that runs the RAG graph and returns a structured result."""

from ..cache import get_cached, set_cached
from ..helpers.logger import get_logger
from .agent_graph import app as agent_graph
from .state import RAGState

logger = get_logger(__name__)


class GraphService:
    """Runs the LangGraph RAG pipeline and returns a structured response."""

    def __init__(self):
        self.graph = agent_graph

    def process_query(self, user_message: str, metadata: dict | None = None) -> dict:
        """Run the RAG pipeline for a user message and return the result.

        Args:
            user_message: The user question to answer.
            metadata: Optional metadata echoed back in the response.

        Returns:
            A dict with ``response``, ``success``, ``error`` and ``metadata``.
            On abstention or empty answer a fallback message is returned; on
            error ``success`` is False and ``error`` holds the message.
        """
        try:
            cached = get_cached(user_message)
        except Exception:
            logger.info("Cache read failed; treating as miss", exc_info=True)
            cached = None
        if cached:
            return {
                "response": cached,
                "success": True,
                "error": None,
                "metadata": metadata or {},
            }
        try:
            initial_state: RAGState = {
                "query": user_message,
                "documents": [],
                "scores": [],
                "answer": "",
                "abstain": False,
            }
            result = self.graph.invoke(initial_state)
            abstain = result.get("abstain")
            answer = result.get("answer")

            if not abstain and answer:
                try:
                    set_cached(user_message, answer)
                except Exception:
                    logger.info("Cache write failed; answer not cached", exc_info=True)

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
        """Return a Mermaid diagram of the graph, or an error string on failure."""
        try:
            return self.graph.get_graph().draw_mermaid()
        except Exception as e:
            return f"Error generando visualización: {e}"


graph_service = GraphService()
