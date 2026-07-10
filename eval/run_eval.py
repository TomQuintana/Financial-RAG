"""RAG evaluation pipeline using deepeval over eval_questions.json."""

import json
import sys
from pathlib import Path

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.graph.agent_graph import app as agent_graph
from src.graph.state import RAGState

QUESTIONS_FILE = Path(__file__).parent / "eval_questions.json"


def build_test_cases() -> list[LLMTestCase]:
    """Build deepeval test cases by running the graph over each eval question.

    Returns:
        A list of LLMTestCase holding the generated answer and retrieval
        context for every question in eval_questions.json.
    """
    questions = json.loads(QUESTIONS_FILE.read_text())
    cases = []
    for q in questions:
        state = agent_graph.invoke(
            RAGState(query=q["question"], documents=[], scores=[], answer="", abstain=False)
        )
        cases.append(
            LLMTestCase(
                input=q["question"],
                actual_output=state["answer"],
                retrieval_context=[d.page_content for d in state["documents"]],
                expected_output=q["ground_truth"] or None,
            )
        )
    return cases


if __name__ == "__main__":
    test_cases = build_test_cases()
    has_ground_truth = any(tc.expected_output for tc in test_cases)

    metrics = [FaithfulnessMetric(), AnswerRelevancyMetric()]
    if has_ground_truth:
        metrics += [ContextualPrecisionMetric(), ContextualRecallMetric()]

    evaluate(test_cases, metrics=metrics)
