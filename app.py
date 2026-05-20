from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from urllib.parse import quote

import streamlit as st
from dotenv import load_dotenv

from rag_workflow.document_loader import (
    SUPPORTED_EXTENSIONS,
    chunk_document,
    iter_supported_files,
    load_file,
)
from rag_workflow.llm import build_augmented_user_message, build_context, chat, citation_list
from rag_workflow.onedrive import download_from_onedrive
from rag_workflow.vector_store import get_collection, query_collection, upsert_chunks


load_dotenv()

DEFAULT_SYSTEM_PROMPT = """You answer questions using only the provided document context.
If the answer is not in the context, say that the documents do not contain enough information.
Be concise and cite source numbers when citations are available."""

DEFAULT_PROMPT_TEMPLATE = """Document context:
{context}

Question:
{question}

Answer using the document context. Include source numbers for factual claims."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_setting(name: str, default: str = "") -> str:
    env_value = os.getenv(name)
    if env_value is not None:
        return env_value
    try:
        value = st.secrets.get(name, default)
    except Exception:
        return default
    return str(value)


def setting_enabled(name: str, default: bool = False) -> bool:
    value = get_setting(name, "true" if default else "false")
    return value.lower() in {"1", "true", "yes", "on"}


def hide_streamlit_deploy_controls() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        #MainMenu {
            display: none !important;
            visibility: hidden !important;
        }
        header {
            visibility: hidden;
        }
        header::after {
            content: "";
            visibility: visible;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Configurable RAG Workflow", layout="wide")
    hide_streamlit_deploy_controls()
    st.title("Configurable RAG Workflow")
    st.caption("Upload documents or ingest OneDrive files, index locally, and choose local or hosted AI backends.")

    history_path = Path(get_setting("RAG_DATA_DIR", ".rag_data/documents")).parent / "chat_history.json"
    load_chat_history(history_path)
    render_recents_sidebar(history_path)
    cfg = sidebar_config()
    cfg["history_path"] = history_path
    render_privacy_notice(cfg)
    try:
        collection = get_collection(
            persist_dir=cfg["chroma_dir"],
            collection_name=cfg["collection_name"],
            embedding_provider=cfg["embedding_provider"],
            embedding_model=cfg["embedding_model"],
            embedding_endpoint=cfg["embedding_endpoint"],
            embedding_api_key=cfg["embedding_api_key"],
            allow_remote_embedding_endpoint=cfg["allow_remote_embedding_endpoint"],
            distance_metric=cfg["distance_metric"],
        )
    except Exception as exc:
        st.error(f"Could not open vector collection: {exc}")
        st.info(
            "The app normally creates a separate collection for each embedding provider/model. "
            "If you set `RAG_COLLECTION` manually, remove it or change it, then re-index your documents."
        )
        st.stop()

    tab_chat, tab_ingest, tab_onedrive, tab_index = st.tabs(
        ["Chat", "Upload & Ingest", "OneDrive", "Index"]
    )

    with tab_chat:
        render_chat_tab(cfg, collection)
    with tab_ingest:
        render_ingest_tab(cfg, collection)
    with tab_onedrive:
        render_onedrive_tab(cfg, collection)
    with tab_index:
        render_index_tab(collection)


def sidebar_config() -> dict:
    with st.sidebar:
        st.header("Embeddings")
        embedding_provider = st.selectbox(
            "Embedding provider",
            ["sentence-transformers", "openai-compatible"],
            index=0 if get_setting("RAG_EMBEDDING_PROVIDER", "openai-compatible") == "sentence-transformers" else 1,
        )
        if embedding_provider == "sentence-transformers":
            embedding_model = st.text_input(
                "Sentence-transformers model",
                get_setting("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            )
            embedding_endpoint = ""
            embedding_api_key = ""
            allow_remote_embedding_endpoint = False
        else:
            embedding_endpoint = st.text_input(
                "Embeddings base URL",
                get_setting("RAG_EMBEDDING_BASE_URL", "https://api.openai.com/v1"),
            )
            embedding_model = st.text_input(
                "Embedding model",
                get_setting("RAG_EMBEDDING_MODEL", "text-embedding-3-small"),
            )
            embedding_api_key = st.text_input(
                "Embedding API key",
                get_setting("RAG_EMBEDDING_API_KEY", ""),
                type="password",
            )
            allow_remote_embedding_endpoint = st.checkbox(
                "Allow non-local embedding endpoint",
                value=setting_enabled("RAG_ALLOW_REMOTE_EMBEDDINGS"),
            )
        distance_metric = st.selectbox("Vector distance metric", ["cosine", "l2", "ip"], index=0)

        data_dir = Path(get_setting("RAG_DATA_DIR", ".rag_data/documents"))
        chroma_dir = Path(get_setting("RAG_CHROMA_DIR", ".rag_data/chroma"))
        collection_name = get_setting(
            "RAG_COLLECTION",
            make_collection_name(embedding_provider=embedding_provider, embedding_model=embedding_model),
        )

        st.header("Chunking")
        chunk_size = st.slider("Chunk size (characters)", 300, 6000, 1200, 100)
        chunk_overlap = st.slider("Chunk overlap (characters)", 0, 1000, 150, 25)
        min_chunk_length = st.slider("Minimum chunk length", 20, 1000, 80, 10)
        replace_sources = st.checkbox("Replace existing chunks for same source", value=True)

        st.header("Retrieval")
        top_k = st.slider("Retrieved chunks", 1, 30, 6)
        use_distance_threshold = st.checkbox("Use distance threshold", value=False)
        distance_threshold = None
        if use_distance_threshold:
            distance_threshold = st.slider("Maximum distance", 0.0, 2.0, 0.8, 0.05)
        dedupe_sources = st.checkbox("Return at most one chunk per source", value=False)
        max_context_chars = st.slider("Max context characters", 1000, 60000, 12000, 1000)

        st.header("Prompting")
        system_prompt = st.text_area("System prompt", DEFAULT_SYSTEM_PROMPT, height=140)
        prompt_template = st.text_area("RAG prompt template", DEFAULT_PROMPT_TEMPLATE, height=180)
        include_citations = st.checkbox("Show retrieved citations", value=True)
        history_turns = st.slider("Chat history turns sent to model", 0, 10, 3)

        st.header("Answer model")
        default_backend = get_setting("RAG_LLM_BACKEND", "openai-compatible")
        backend = st.selectbox(
            "Backend",
            ["openai-compatible", "ollama"],
            index=0 if default_backend == "openai-compatible" else 1,
        )
        if backend == "ollama":
            endpoint = st.text_input("Ollama host", get_setting("RAG_OLLAMA_HOST", "http://localhost:11434"))
            model = st.text_input("Ollama model", get_setting("RAG_OLLAMA_MODEL", "llama3:latest"))
            api_key = ""
        else:
            endpoint = st.text_input("Chat base URL", get_setting("RAG_OPENAI_COMPAT_BASE_URL", "https://api.openai.com/v1"))
            model = st.text_input("Chat model", get_setting("RAG_OPENAI_COMPAT_MODEL", "gpt-4o-mini"))
            api_key = st.text_input("API key (optional)", get_setting("RAG_OPENAI_COMPAT_API_KEY", ""), type="password")
        allow_remote_endpoint = st.checkbox(
            "Allow non-local chat endpoint",
            value=setting_enabled("RAG_ALLOW_REMOTE_CHAT"),
        )

        with st.expander("Generation parameters", expanded=True):
            temperature = st.slider("Temperature", 0.0, 2.0, 0.2, 0.05)
            top_p = st.slider("Top-p", 0.0, 1.0, 0.9, 0.05)
            top_k_model = st.slider("Top-k (Ollama)", 1, 200, 40)
            repeat_penalty = st.slider("Repeat penalty (Ollama)", 0.5, 2.0, 1.1, 0.05)
            num_ctx = st.slider("Context window (Ollama)", 1024, 131072, 8192, 1024)
            num_predict = st.slider("Max output tokens", 128, 8192, 1024, 128)
            seed_text = st.text_input("Seed (blank for random)", "")
            stop_text = st.text_input("Stop sequences (comma-separated)", "")

    data_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    return {
        "data_dir": data_dir,
        "chroma_dir": chroma_dir,
        "collection_name": collection_name,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "embedding_endpoint": embedding_endpoint,
        "embedding_api_key": embedding_api_key,
        "allow_remote_embedding_endpoint": allow_remote_embedding_endpoint,
        "distance_metric": distance_metric,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "min_chunk_length": min_chunk_length,
        "replace_sources": replace_sources,
        "top_k": top_k,
        "distance_threshold": distance_threshold,
        "dedupe_sources": dedupe_sources,
        "max_context_chars": max_context_chars,
        "system_prompt": system_prompt,
        "prompt_template": prompt_template,
        "include_citations": include_citations,
        "history_turns": history_turns,
        "backend": backend,
        "endpoint": endpoint,
        "model": model,
        "api_key": api_key,
        "allow_remote_endpoint": allow_remote_endpoint,
        "temperature": temperature,
        "top_p": top_p,
        "top_k_model": top_k_model,
        "repeat_penalty": repeat_penalty,
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        "seed": int(seed_text) if seed_text.strip() else None,
        "stop_sequences": [item.strip() for item in stop_text.split(",") if item.strip()],
    }


def render_privacy_notice(cfg: dict) -> None:
    remote_embeddings = cfg["embedding_provider"] == "openai-compatible" and cfg["allow_remote_embedding_endpoint"]
    remote_chat = cfg["backend"] == "openai-compatible" and cfg["allow_remote_endpoint"]
    if remote_embeddings and remote_chat:
        st.warning(
            "Cloud-assisted mode is enabled: indexed document chunks are sent to the embedding endpoint, "
            "and retrieved chunks are sent to the chat endpoint for each question."
        )
    elif remote_embeddings:
        st.warning("Remote embeddings are enabled: every indexed document chunk is sent to the embedding endpoint.")
    elif remote_chat:
        st.warning("Remote chat is enabled: only the retrieved chunks used for an answer are sent to the chat endpoint.")
    else:
        st.info("Remote data transmission is disabled. Enable a remote checkbox only when you intend to send data to that endpoint.")

    st.caption("When changing embedding provider or embedding model, re-index your documents so the vectors match the selected model.")


def render_chat_tab(cfg: dict, collection) -> None:
    history_path = cfg["history_path"]
    current_chat = get_current_chat()
    messages_for_chat = current_chat["messages"]

    st.subheader(current_chat["title"])

    pending_question = st.session_state.pop("pending_question", None)
    if pending_question:
        if current_chat["title"] == "New chat":
            current_chat["title"] = make_chat_title(pending_question)
        messages_for_chat.append({"role": "user", "content": pending_question})
        current_chat["updated_at"] = utc_now()
        save_chat_history(history_path)

    for message in messages_for_chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if pending_question:
        answer = answer_question(pending_question, cfg, collection, messages_for_chat)
        messages_for_chat.append({"role": "assistant", "content": answer})
        current_chat["updated_at"] = utc_now()
        save_chat_history(history_path)

    # Keep the message box after all rendered messages so it stays at the bottom of the conversation.
    question = st.chat_input("Ask a question about your documents")
    if question:
        st.session_state.pending_question = question
        st.rerun()


def answer_question(question: str, cfg: dict, collection, messages_for_chat: list[dict]) -> str:
    matches = query_collection(
        collection,
        question,
        top_k=cfg["top_k"],
        distance_threshold=cfg["distance_threshold"],
        dedupe_sources=cfg["dedupe_sources"],
    )
    context = build_context(matches, max_context_chars=cfg["max_context_chars"])
    augmented = build_augmented_user_message(
        question=question,
        context=context,
        prompt_template=cfg["prompt_template"],
    )

    history = []
    if cfg["history_turns"] > 0:
        history = messages_for_chat[-cfg["history_turns"] * 2 : -1]

    messages = [{"role": "system", "content": cfg["system_prompt"]}, *history, {"role": "user", "content": augmented}]

    with st.chat_message("assistant"):
        try:
            answer = chat(
                backend=cfg["backend"],
                messages=messages,
                model=cfg["model"],
                endpoint=cfg["endpoint"],
                api_key=cfg["api_key"],
                allow_remote_endpoint=cfg["allow_remote_endpoint"],
                temperature=cfg["temperature"],
                top_p=cfg["top_p"],
                top_k=cfg["top_k_model"],
                repeat_penalty=cfg["repeat_penalty"],
                num_ctx=cfg["num_ctx"],
                num_predict=cfg["num_predict"],
                seed=cfg["seed"],
                stop_sequences=cfg["stop_sequences"],
            )
        except Exception as exc:
            answer = f"Error: {exc}"
        st.markdown(answer)
        if cfg["include_citations"] and matches:
            with st.expander("Retrieved sources"):
                for citation in citation_list(matches):
                    st.write(citation)
    return answer


def load_chat_history(path: Path) -> None:
    if "chat_history_loaded" in st.session_state:
        sync_active_chat_from_query()
        return

    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            chats = payload.get("chats", [])
            if chats:
                st.session_state.chat_sessions = chats
                active_id = payload.get("active_chat_id") or chats[0]["id"]
                st.session_state.active_chat_id = active_id
                st.session_state.chat_history_loaded = True
                sync_active_chat_from_query()
                return
        except Exception:
            pass

    first_chat = new_chat()
    st.session_state.chat_sessions = [first_chat]
    st.session_state.active_chat_id = first_chat["id"]
    st.session_state.chat_history_loaded = True
    sync_active_chat_from_query()
    save_chat_history(path)


def save_chat_history(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "active_chat_id": st.session_state.active_chat_id,
        "chats": st.session_state.chat_sessions,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def new_chat() -> dict:
    now = utc_now()
    return {
        "id": str(uuid.uuid4()),
        "title": "New chat",
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }


def get_current_chat() -> dict:
    active_id = st.session_state.active_chat_id
    for chat_session in st.session_state.chat_sessions:
        if chat_session["id"] == active_id:
            return chat_session

    fallback = st.session_state.chat_sessions[0]
    st.session_state.active_chat_id = fallback["id"]
    return fallback


def render_recents_sidebar(path: Path) -> None:
    sorted_chats = sorted(
        st.session_state.chat_sessions,
        key=lambda chat_session: chat_session.get("updated_at", ""),
        reverse=True,
    )

    st.sidebar.markdown(
        """
        <style>
        section[data-testid="stSidebar"] .recents-title {
            color: #ffffff;
            font-size: 0.92rem;
            font-weight: 700;
            margin: 0.25rem 0 0.45rem 0;
        }
        section[data-testid="stSidebar"] .recent-chat {
            display: block;
            color: #ffffff;
            text-decoration: none;
            padding: 0.55rem 0.7rem;
            margin: 0.08rem 0;
            border-radius: 0.45rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            font-size: 0.92rem;
            line-height: 1.2;
        }
        section[data-testid="stSidebar"] .recent-chat:hover {
            background: rgba(255, 255, 255, 0.09);
            color: #ffffff;
            text-decoration: none;
        }
        section[data-testid="stSidebar"] .recent-chat.active {
            background: rgba(255, 255, 255, 0.18);
        }
        section[data-testid="stSidebar"] .recents-wrap {
            margin-bottom: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if st.sidebar.button("New chat", use_container_width=True):
        chat_session = new_chat()
        st.session_state.chat_sessions.append(chat_session)
        st.session_state.active_chat_id = chat_session["id"]
        save_chat_history(path)
        st.query_params["chat_id"] = chat_session["id"]
        st.rerun()

    links = ['<div class="recents-wrap">', '<div class="recents-title">Recents</div>']
    active_id = st.session_state.active_chat_id
    for chat_session in sorted_chats[:30]:
        title = escape(chat_session.get("title") or "Untitled chat")
        active_class = " active" if chat_session["id"] == active_id else ""
        href = f"?chat_id={quote(chat_session['id'])}"
        links.append(f'<a class="recent-chat{active_class}" href="{href}" title="{title}">{title}</a>')
    links.append("</div>")
    st.sidebar.markdown("\n".join(links), unsafe_allow_html=True)

    current_chat = get_current_chat()
    with st.sidebar.expander("Current chat options", expanded=False):
        rename_value = st.text_input(
            "Title",
            value=current_chat["title"],
            key=f"chat_title_{current_chat['id']}",
        )
        if st.button("Rename", use_container_width=True):
            current_chat["title"] = rename_value.strip() or "Untitled chat"
            current_chat["updated_at"] = utc_now()
            save_chat_history(path)
            st.rerun()
        if st.button("Clear messages", use_container_width=True):
            current_chat["messages"] = []
            current_chat["updated_at"] = utc_now()
            save_chat_history(path)
            st.rerun()
        delete_disabled = len(st.session_state.chat_sessions) <= 1
        if st.button("Delete chat", disabled=delete_disabled, use_container_width=True):
            st.session_state.chat_sessions = [
                chat_session
                for chat_session in st.session_state.chat_sessions
                if chat_session["id"] != current_chat["id"]
            ]
            st.session_state.active_chat_id = st.session_state.chat_sessions[0]["id"]
            save_chat_history(path)
            st.query_params["chat_id"] = st.session_state.active_chat_id
            st.rerun()


def sync_active_chat_from_query() -> None:
    chat_id = st.query_params.get("chat_id")
    if not chat_id:
        return
    chat_ids = {chat_session["id"] for chat_session in st.session_state.chat_sessions}
    if chat_id in chat_ids:
        st.session_state.active_chat_id = chat_id


def make_chat_title(question: str) -> str:
    title = " ".join(question.strip().split())
    if not title:
        return "Untitled chat"
    return title[:60] + ("..." if len(title) > 60 else "")


def make_collection_name(*, embedding_provider: str, embedding_model: str) -> str:
    raw = f"documents_{embedding_provider}_{embedding_model}"
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw).strip("_").lower()
    return name[:63] or "documents"


def render_ingest_tab(cfg: dict, collection) -> None:
    st.subheader("Upload documents")
    uploads = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=[ext.lstrip(".") for ext in sorted(SUPPORTED_EXTENSIONS)],
    )
    upload_dir = cfg["data_dir"] / "uploads"
    if uploads and st.button("Save uploaded files and index"):
        saved_paths = []
        upload_dir.mkdir(parents=True, exist_ok=True)
        for upload in uploads:
            target = upload_dir / upload.name
            target.write_bytes(upload.getbuffer())
            saved_paths.append(target)
        count = ingest_paths(saved_paths, cfg, collection)
        st.success(f"Indexed {count} chunks from {len(saved_paths)} uploaded files.")

    st.divider()
    st.subheader("Ingest local folder")
    folder = st.text_input("Local folder path", "")
    recursive = st.checkbox("Include subfolders", value=True)
    if st.button("Index local folder"):
        root = Path(folder).expanduser()
        if not root.exists():
            st.error(f"Folder does not exist: {root}")
            return
        paths = list(iter_supported_files(root)) if recursive else [
            path for path in root.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        count = ingest_paths(paths, cfg, collection)
        st.success(f"Indexed {count} chunks from {len(paths)} files.")


def render_onedrive_tab(cfg: dict, collection) -> None:
    st.subheader("OneDrive local synced folder")
    st.write("Point this at a local OneDrive folder if the OneDrive desktop client is already syncing your files.")
    local_onedrive = st.text_input("Synced OneDrive folder path", "")
    if st.button("Index synced OneDrive folder"):
        root = Path(local_onedrive).expanduser()
        if not root.exists():
            st.error(f"Folder does not exist: {root}")
            return
        paths = list(iter_supported_files(root))
        count = ingest_paths(paths, cfg, collection)
        st.success(f"Indexed {count} chunks from {len(paths)} OneDrive-synced files.")

    st.divider()
    st.subheader("OneDrive Microsoft Graph connector")
    st.write("Downloads matching files from OneDrive to local storage, then indexes them locally.")
    client_id = st.text_input("Client ID", get_setting("ONEDRIVE_CLIENT_ID", ""))
    tenant_id = st.text_input("Tenant ID", get_setting("ONEDRIVE_TENANT_ID", "common"))
    remote_path = st.text_input("Remote folder path in OneDrive", "")
    max_files = st.slider("Maximum files to download", 1, 1000, 100)
    extensions = st.multiselect(
        "File extensions",
        sorted(SUPPORTED_EXTENSIONS),
        default=sorted(SUPPORTED_EXTENSIONS),
    )
    if st.button("Download from OneDrive Graph and index"):
        if not client_id:
            st.error("Client ID is required for Microsoft Graph.")
            return
        destination = cfg["data_dir"] / "onedrive_graph"
        token_cache = cfg["data_dir"].parent / "msal_token_cache.json"
        device_placeholder = st.empty()

        def show_device_flow(flow: dict) -> None:
            device_placeholder.info(
                "Open "
                f"{flow.get('verification_uri', 'the Microsoft device login page')} "
                f"and enter code: {flow.get('user_code', '')}"
            )

        with st.spinner("Waiting for Microsoft device login if needed, then downloading files..."):
            try:
                paths = download_from_onedrive(
                    client_id=client_id,
                    tenant_id=tenant_id,
                    remote_path=remote_path,
                    destination_dir=destination,
                    extensions=set(extensions),
                    max_files=max_files,
                    token_cache_path=token_cache,
                    device_flow_callback=show_device_flow,
                )
            except Exception as exc:
                st.error(f"OneDrive download failed: {exc}")
                return
        device_placeholder.empty()
        count = ingest_paths(paths, cfg, collection)
        st.success(f"Downloaded {len(paths)} files and indexed {count} chunks.")


def render_index_tab(collection) -> None:
    st.subheader("Index status")
    try:
        count = collection.count()
    except Exception as exc:
        st.error(f"Unable to read index: {exc}")
        return
    st.metric("Chunks in collection", count)
    if st.button("Show sample indexed chunks"):
        sample = collection.peek(limit=10)
        for doc, metadata in zip(sample.get("documents", []), sample.get("metadatas", [])):
            with st.expander(metadata.get("file_name", "unknown")):
                st.write(metadata)
                st.text(doc[:1500])

    st.divider()
    st.subheader("Danger zone")
    st.write("This deletes the local vector collection only. It does not delete source documents.")
    if st.button("Delete all indexed chunks"):
        existing = collection.get(include=[])
        ids = existing.get("ids", [])
        if ids:
            collection.delete(ids=ids)
        st.success("Index cleared.")


def ingest_paths(paths: list[Path], cfg: dict, collection) -> int:
    total = 0
    progress = st.progress(0)
    status = st.empty()
    paths = [path for path in paths if path.suffix.lower() in SUPPORTED_EXTENSIONS]
    for index, path in enumerate(paths, start=1):
        status.write(f"Indexing {path.name} ({index}/{len(paths)})")
        try:
            document = load_file(path)
            chunks = chunk_document(
                document,
                chunk_size=cfg["chunk_size"],
                chunk_overlap=cfg["chunk_overlap"],
                min_chunk_length=cfg["min_chunk_length"],
            )
            total += upsert_chunks(collection, chunks, replace_sources=cfg["replace_sources"])
        except Exception as exc:
            st.warning(f"Skipped {path}: {exc}")
        progress.progress(index / max(len(paths), 1))
    status.empty()
    return total


if __name__ == "__main__":
    main()
