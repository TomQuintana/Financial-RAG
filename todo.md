# Bitácora

## Limpieza pendiente — PR #1 (FIN-14)

Solo cosmético, no cambia comportamiento (`net: -9 líneas`):

- [ ] `abstain_agent.py` — quitar temp `is_abstein`: `return not scores or max(scores) <= RELEVANCE_THRESHOLD`
- [ ] `generate_output_agent.py` L11 — borrar `# context = ...` comentado
- [ ] `agent_graph.py` L11 — borrar `# return END if ...` comentado
- [ ] `agent_graph.py` `route_after_abstain` — volver a una línea: `return END if state.get("abstain") else "generate"`
- [ ] `agent_graph.py` — quitar 3 líneas en blanco agregadas entre los `add_edge`
- [ ] `generate_output_prompt.py` — inline: `context = "\n\n".join(_format_doc(i, d) for i, d in enumerate(documents, 1))`
- [ ] Centralizar mensajes y constantes en `src/constants/` (o similar): strings como `"No encontré información relevante en los reportes financieros."`, `"No se generó respuesta."` y constantes como `RELEVANCE_THRESHOLD = 0.0` hoy están hardcodeadas dispersas

## Pendiente — Observabilidad

- [ ] Integrar Langfuse o LangSmith para tracing de prompts/respuestas y evals en producción
- [ ] Pasar (loggear) los prompts reales al proveedor LLM que se use, para poder inspeccionarlos en el dashboard de tracing

## 2026-06-28

### `src/ingest.py` — Steps 1, 2 y 3

**Extracción:**
- Fix iXBRL: supresión `XMLParsedAsHTMLWarning` + remoción `<ix:header>` (evita basura XBRL en texto extraído)
- Ticker parsing con `-` (EDGAR naming: `aapl-20250927.html`)
- 4 reportes: AAPL (29k words), JPM (174k), KO (90k), PFE (92k)

**Chunking:**
- Import directo `langchain_text_splitters.character` (evita cadena pesada vía `sentence_transformers`)
- `CHUNK_SIZE=3200` chars / `CHUNK_OVERLAP=400`
- 1,108 chunks totales, avg ~2,400 chars/chunk

**Embeddings:**
- `build_embeddings()` con `text-embedding-3-small`
- Persistencia en `data/chroma_db/`, IDs únicos `AAPL_0042`
