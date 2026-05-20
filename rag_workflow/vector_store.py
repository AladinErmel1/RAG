from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import requests

from rag_workflow.document_loader import TextChunk


LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


class OpenAICompatibleEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(
        self,
        *,
        model: str,
        endpoint: str,
        api_key: str | None,
        allow_remote_endpoint: bool,
    ) -> None:
        if not allow_remote_endpoint and not _is_local_url(endpoint):
            raise ValueError(
                f"Embedding endpoint {endpoint!r} is not local. Enable remote embeddings only if you intend to send chunks there."
            )
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key

    def __call__(self, input: Documents) -> Embeddings:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(
            f"{self.endpoint}/embeddings",
            headers=headers,
            data=json.dumps({"model": self.model, "input": input}),
            timeout=600,
        )
        response.raise_for_status()
        payload = response.json()
        return [item["embedding"] for item in payload["data"]]

    def name(self) -> str:
        return "openai-compatible"

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine", "l2", "ip"]

    def get_config(self) -> dict:
        return {
            "model": self.model,
            "endpoint": self.endpoint,
            "api_key": "redacted" if self.api_key else "",
        }

    @staticmethod
    def build_from_config(config: dict) -> "OpenAICompatibleEmbeddingFunction":
        return OpenAICompatibleEmbeddingFunction(
            model=config["model"],
            endpoint=config["endpoint"],
            api_key=None,
            allow_remote_endpoint=True,
        )


def get_collection(
    *,
    persist_dir: Path,
    collection_name: str,
    embedding_provider: str,
    embedding_model: str,
    embedding_endpoint: str | None = None,
    embedding_api_key: str | None = None,
    allow_remote_embedding_endpoint: bool = False,
    distance_metric: str = "cosine",
):
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    if embedding_provider == "sentence-transformers":
        embedding_function = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
    elif embedding_provider == "openai-compatible":
        if not embedding_endpoint:
            raise ValueError("Embedding endpoint is required for OpenAI-compatible embeddings.")
        embedding_function = OpenAICompatibleEmbeddingFunction(
            model=embedding_model,
            endpoint=embedding_endpoint,
            api_key=embedding_api_key,
            allow_remote_endpoint=allow_remote_embedding_endpoint,
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {embedding_provider}")

    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": distance_metric},
    )


def _is_local_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.hostname in LOCAL_HOSTS


def upsert_chunks(collection, chunks: list[TextChunk], *, replace_sources: bool) -> int:
    if not chunks:
        return 0

    if replace_sources:
        for source in sorted({chunk.metadata["source"] for chunk in chunks}):
            try:
                collection.delete(where={"source": source})
            except Exception:
                pass

    collection.upsert(
        ids=[chunk.id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
    )
    return len(chunks)


def query_collection(
    collection,
    query: str,
    *,
    top_k: int,
    distance_threshold: float | None,
    dedupe_sources: bool,
) -> list[dict]:
    results = collection.query(
        query_texts=[query],
        n_results=top_k * 3 if dedupe_sources else top_k,
        include=["documents", "metadatas", "distances"],
    )
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    seen_sources: set[str] = set()
    matches: list[dict] = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        if distance_threshold is not None and distance > distance_threshold:
            continue
        source = metadata.get("source", "")
        if dedupe_sources and source in seen_sources:
            continue
        seen_sources.add(source)
        matches.append(
            {
                "text": text,
                "metadata": metadata,
                "distance": distance,
            }
        )
        if len(matches) >= top_k:
            break
    return matches
