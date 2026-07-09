from typing import cast

from langgraph.graph import END

from src.graph.agent_graph import route_after_abstain
from src.graph.state import RAGState


def make_state(**overrides) -> RAGState:
    return cast(
        RAGState,
        {
            "query": "",
            "documents": [],
            "scores": [],
            "answer": "",
            "abstain": False,
            **overrides,
        },
    )


def test_route_retorna_end_cuando_abstain_true():
    assert route_after_abstain(make_state(abstain=True)) == END


def test_route_retorna_generate_cuando_abstain_false():
    assert route_after_abstain(make_state(abstain=False)) == "generate"
