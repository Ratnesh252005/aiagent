from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from rapidfuzz import fuzz

# Types expected from existing codebase
# - EmbeddingGenerator with method generate_single_embedding(text: str) -> np.ndarray
# - PineconeVectorStore with method search_similar(query_embedding: np.ndarray, top_k: int, document_id: Optional[str]) -> List[Dict[str, Any]]


@dataclass
class RetrievedChunk:
    id: str
    text: str
    chunk_number: int
    score: float
    lex_score: float
    final_score: float
    matched_query: str
    base_query: str
    document_id: Optional[str] = None
    timestamp: Optional[str] = None


class RetrieverAgent:
    """
    Retriever Agent that encapsulates query embedding, vector search, de-duplication,
    and hybrid re-ranking (vector score + lexical score).
    """

    def __init__(self, embedding_generator, vector_store):
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store

    def retrieve(
        self,
        questions: List[str],
        top_k: int,
        document_id: Optional[str],
        base_query: Optional[str] = None,
        vector_weight: float = 0.7,
        lexical_weight: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Perform retrieval for one or more questions and return top reranked chunks.

        Args:
            questions: list of sub-questions or a single-element list with the main query
            top_k: number of results to return after re-ranking
            document_id: filter results to a specific document if provided
            base_query: the primary query (used for diagnostics display); if None, uses each question
            vector_weight: weight for vector similarity score
            lexical_weight: weight for lexical similarity score

        Returns:
            List of dicts with fields: id, text, chunk_number, score, lex_score, final_score,
            matched_query/base_query, document_id, timestamp
        """
        if not questions:
            return []

        aggregated: List[Dict[str, Any]] = []
        seen_ids = set()

        # 1) Vector retrieval per question (batched embeddings for performance)
        try:
            embeds = self.embedding_generator.generate_embeddings_quiet(questions)
        except Exception:
            embeds = []

        for q, q_embed in zip(questions, embeds):
            if q_embed is None or not isinstance(q_embed, np.ndarray) or getattr(q_embed, "size", 0) == 0:
                continue

            results = self.vector_store.search_similar(
                q_embed,
                top_k=top_k,
                document_id=document_id,
            )

            for r in results:
                rid = r.get("id")
                if not rid or rid in seen_ids:
                    continue
                seen_ids.add(rid)

                item = dict(r)
                item["matched_query"] = q
                # Ensure required fields exist
                item.setdefault("text", "")
                item.setdefault("score", 0.0)
                item.setdefault("chunk_number", 0)
                aggregated.append(item)

        if not aggregated:
            return []

        # 2) Hybrid re-ranking
        reranked: List[Dict[str, Any]] = []
        for ch in aggregated:
            vec_score = float(ch.get("score", 0.0))
            q_for_lex = base_query or ch.get("matched_query", "")
            lex_score = fuzz.token_set_ratio(q_for_lex, ch.get("text", "")) / 100.0
            final_score = vector_weight * vec_score + lexical_weight * lex_score

            item = dict(ch)
            item["lex_score"] = lex_score
            item["final_score"] = final_score
            item["base_query"] = q_for_lex
            reranked.append(item)

        reranked.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        top_context = reranked[: top_k]
        return top_context
