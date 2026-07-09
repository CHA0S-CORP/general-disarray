"""
Knowledge Base (RAG)
====================
Indexes text/markdown documents from data/knowledge/ into a small local
vector store and answers similarity queries for the KNOWLEDGE tool (and the
optional auto-inject path).

Embeddings run locally on CPU via fastembed (ONNX); the index is a
langchain-core InMemoryVectorStore persisted to data/knowledge_index/ so a
restart with unchanged documents skips re-embedding. Everything is fail-open:
missing dependencies, an empty directory, or an indexing error just mean the
knowledge base reports itself unavailable/not-ready.
"""

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

from logging_utils import log_event

try:
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    KNOWLEDGE_DEPS_AVAILABLE = True
except ImportError:
    Embeddings = object  # type: ignore[assignment,misc]
    KNOWLEDGE_DEPS_AVAILABLE = False

logger = logging.getLogger(__name__)

# File types indexed from the knowledge directory.
_INDEXABLE_SUFFIXES = (".txt", ".md", ".markdown")


class _FastEmbedEmbeddings(Embeddings):
    """Minimal langchain Embeddings adapter over fastembed.TextEmbedding.

    Kept local to avoid depending on langchain-community for one class.
    """

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding  # heavy import, deferred
        cache_dir = os.getenv("FASTEMBED_CACHE_PATH") or None
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [vec.tolist() for vec in self._model.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class KnowledgeBase:
    """Document index over the knowledge directory.

    Lifecycle: constructed cheaply (no model loading); `start()` builds or
    loads the index in a worker thread. `available` says whether the feature
    can work at all (enabled + deps + documents present); `ready` says the
    index is currently queryable.
    """

    def __init__(self, config, embeddings=None):
        self.config = config
        self._embeddings = embeddings  # injectable for tests
        self._store = None
        self._index_dir: Path = config.data_dir / "knowledge_index"
        self.ready = False

    # ---- status ------------------------------------------------------------

    def _documents(self) -> List[Path]:
        knowledge_dir = self.config.knowledge_dir
        if not knowledge_dir or not Path(knowledge_dir).is_dir():
            return []
        return sorted(
            p for p in Path(knowledge_dir).rglob("*")
            if p.is_file() and p.suffix.lower() in _INDEXABLE_SUFFIXES)

    @property
    def available(self) -> bool:
        """Feature can work: enabled, deps importable, documents present."""
        if not self.config.knowledge_enabled or not KNOWLEDGE_DEPS_AVAILABLE:
            return False
        if self._embeddings is None:
            import importlib.util
            if importlib.util.find_spec("fastembed") is None:
                return False
        return bool(self._documents())

    # ---- indexing ----------------------------------------------------------

    def _fingerprint(self, docs: List[Path]) -> str:
        h = hashlib.sha256()
        h.update(f"{self.config.knowledge_chunk_size}:"
                 f"{self.config.knowledge_chunk_overlap}:"
                 f"{self.config.knowledge_embedding_model}".encode())
        for p in docs:
            stat = p.stat()
            h.update(f"{p}:{stat.st_mtime_ns}:{stat.st_size}".encode())
        return h.hexdigest()

    async def start(self) -> None:
        """Build or load the index off the event loop. Never raises."""
        if not self.available:
            if self.config.knowledge_enabled and not KNOWLEDGE_DEPS_AVAILABLE:
                logger.info("Knowledge base disabled: langchain/fastembed not installed")
            return
        try:
            await asyncio.to_thread(self._index)
        except Exception as e:
            logger.error(f"Knowledge indexing failed: {e}")

    def _index(self) -> None:
        docs = self._documents()
        if not docs:
            return
        fingerprint = self._fingerprint(docs)
        self._index_dir.mkdir(parents=True, exist_ok=True)
        index_path = self._index_dir / "index.json"
        fp_path = self._index_dir / "fingerprint.json"

        if self._embeddings is None:
            self._embeddings = _FastEmbedEmbeddings(
                self.config.knowledge_embedding_model)

        # Unchanged corpus -> load the persisted index, skip re-embedding.
        if index_path.exists() and fp_path.exists():
            try:
                stored = json.loads(fp_path.read_text()).get("fingerprint")
                if stored == fingerprint:
                    self._store = InMemoryVectorStore.load(
                        str(index_path), self._embeddings)
                    self.ready = True
                    logger.info(
                        f"Knowledge index loaded from cache ({len(docs)} documents)")
                    return
            except Exception as e:
                logger.warning(f"Could not load cached knowledge index: {e}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.knowledge_chunk_size,
            chunk_overlap=self.config.knowledge_chunk_overlap)
        texts: List[str] = []
        metadatas: List[dict] = []
        for doc in docs:
            try:
                content = doc.read_text(errors="replace")
            except Exception as e:
                logger.warning(f"Skipping unreadable knowledge file {doc}: {e}")
                continue
            for chunk in splitter.split_text(content):
                texts.append(chunk)
                metadatas.append({"source": doc.name})
        if not texts:
            return

        store = InMemoryVectorStore(self._embeddings)
        store.add_texts(texts, metadatas=metadatas)
        self._store = store
        self.ready = True
        try:
            store.dump(str(index_path))
            fp_path.write_text(json.dumps({"fingerprint": fingerprint}))
        except Exception as e:
            logger.warning(f"Could not persist knowledge index: {e}")
        log_event(logger, logging.INFO,
                  f"Knowledge base indexed: {len(docs)} documents, "
                  f"{len(texts)} chunks",
                  event="knowledge_index", documents=len(docs), chunks=len(texts))

    # ---- retrieval ---------------------------------------------------------

    async def search(self, query: str, k: Optional[int] = None
                     ) -> List[Tuple[str, str]]:
        """Top-k (source, chunk) pairs for a query; [] when not ready."""
        if not self.ready or self._store is None or not query.strip():
            return []
        k = k or self.config.knowledge_top_k
        try:
            # Query embedding is a CPU ONNX pass — keep it off the event loop.
            results = await asyncio.to_thread(
                self._store.similarity_search, query, k)
        except Exception as e:
            logger.warning(f"Knowledge search failed: {e}")
            return []
        return [(doc.metadata.get("source", "unknown"), doc.page_content)
                for doc in results]

    async def format_for_prompt(self, query: str) -> str:
        """Render top-k chunks as a prompt block ('' when nothing to add)."""
        results = await self.search(query)
        return "\n\n".join(
            f"[{source}]\n{chunk}" for source, chunk in results)
