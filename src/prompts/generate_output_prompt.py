def generate_output_prompt(query: str, context: str) -> str:
    return (
        f"Answer the question using only the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )
