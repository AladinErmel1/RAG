from __future__ import annotations

import json
from typing import Iterable
from urllib.parse import urlparse

import requests


LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def is_local_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.hostname in LOCAL_HOSTS


def build_context(matches: list[dict], *, max_context_chars: int) -> str:
    parts: list[str] = []
    used = 0
    for index, match in enumerate(matches, start=1):
        metadata = match["metadata"]
        source = metadata.get("file_name") or metadata.get("source", "unknown")
        chunk_index = metadata.get("chunk_index", "?")
        header = f"[Source {index}: {source}, chunk {chunk_index}]"
        body = match["text"].strip()
        block = f"{header}\n{body}"
        if used + len(block) > max_context_chars:
            remaining = max_context_chars - used
            if remaining <= len(header) + 100:
                break
            block = block[:remaining]
        parts.append(block)
        used += len(block)
        if used >= max_context_chars:
            break
    return "\n\n---\n\n".join(parts)


def build_augmented_user_message(
    *,
    question: str,
    context: str,
    prompt_template: str,
) -> str:
    return prompt_template.format(question=question, context=context)


def chat(
    *,
    backend: str,
    messages: list[dict],
    model: str,
    endpoint: str,
    api_key: str | None,
    allow_remote_endpoint: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repeat_penalty: float,
    num_ctx: int,
    num_predict: int,
    seed: int | None,
    stop_sequences: list[str],
) -> str:
    if not allow_remote_endpoint and not is_local_url(endpoint):
        raise ValueError(
            f"Endpoint {endpoint!r} is not local. Enable remote endpoints only if you intend to send prompts there."
        )

    if backend == "ollama":
        return _ollama_chat(
            messages=messages,
            model=model,
            endpoint=endpoint,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            num_ctx=num_ctx,
            num_predict=num_predict,
            seed=seed,
            stop_sequences=stop_sequences,
        )

    if backend == "openai-compatible":
        return _openai_compatible_chat(
            messages=messages,
            model=model,
            endpoint=endpoint,
            api_key=api_key,
            temperature=temperature,
            top_p=top_p,
            num_predict=num_predict,
            seed=seed,
            stop_sequences=stop_sequences,
        )

    raise ValueError(f"Unsupported backend: {backend}")


def _ollama_chat(
    *,
    messages: list[dict],
    model: str,
    endpoint: str,
    temperature: float,
    top_p: float,
    top_k: int,
    repeat_penalty: float,
    num_ctx: int,
    num_predict: int,
    seed: int | None,
    stop_sequences: list[str],
) -> str:
    options = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repeat_penalty": repeat_penalty,
        "num_ctx": num_ctx,
        "num_predict": num_predict,
    }
    if seed is not None:
        options["seed"] = seed
    if stop_sequences:
        options["stop"] = stop_sequences

    response = requests.post(
        f"{endpoint.rstrip('/')}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options,
        },
        timeout=600,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("message", {}).get("content", "")


def _openai_compatible_chat(
    *,
    messages: list[dict],
    model: str,
    endpoint: str,
    api_key: str | None,
    temperature: float,
    top_p: float,
    num_predict: int,
    seed: int | None,
    stop_sequences: list[str],
) -> str:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": num_predict,
    }
    if seed is not None:
        payload["seed"] = seed
    if stop_sequences:
        payload["stop"] = stop_sequences

    response = requests.post(
        f"{endpoint.rstrip('/')}/chat/completions",
        headers=headers,
        data=json.dumps(payload),
        timeout=600,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"]["content"]


def citation_list(matches: list[dict]) -> Iterable[str]:
    for index, match in enumerate(matches, start=1):
        metadata = match["metadata"]
        file_name = metadata.get("file_name", "unknown")
        chunk = metadata.get("chunk_index", "?")
        distance = match.get("distance")
        if distance is None:
            yield f"[{index}] {file_name}, chunk {chunk}"
        else:
            yield f"[{index}] {file_name}, chunk {chunk}, distance {distance:.4f}"
