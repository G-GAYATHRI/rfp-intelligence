from functools import lru_cache
from sentence_transformers import SentenceTransformer
import config

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(config.EMBED_MODEL)


def embed_documents(texts: list[str]) -> list[list[float]]:
    return _model().encode(texts, normalize_embeddings=True,
                           batch_size=32, show_progress_bar=False).tolist()


def embed_query(query: str) -> list[float]:
    return _model().encode(QUERY_PREFIX + query,
                           normalize_embeddings=True).tolist()