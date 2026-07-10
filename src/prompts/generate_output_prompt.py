"""Prompt builder for grounded, source-cited answer generation."""

from langchain_core.documents import Document


def _format_doc(i: int, doc: Document) -> str:
    company = doc.metadata.get("company", "?")
    chunk = doc.metadata.get("chunk_index", "?")
    return f"[{i}] {company} (chunk {chunk}):\n{doc.page_content}"


def generate_output_prompt(query: str, documents: list[Document]) -> str:
    """Build the answer prompt from a query and its context documents.

    Args:
        query: The user question.
        documents: Context documents, rendered with source citations.

    Returns:
        The full prompt string instructing the LLM to answer from context only.
    """
    indexed_docs = list(enumerate(documents, 1))

    formatted_blocks = [_format_doc(i, d) for i, d in indexed_docs]

    context = "\n\n".join(formatted_blocks)

    return (
        f"Answer the question using only the context below. "
        f"If the answer is not in the context, say you don't know. "
        f"Cite the source as [Company, chunk N] after each claim.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )
