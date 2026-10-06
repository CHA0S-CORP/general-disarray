"""Unit tests for the RAG knowledge base.

Uses a deterministic hash-based fake embedder so no ONNX model is ever
downloaded — fastembed is never imported in the test tier.
"""
import hashlib
import re
from typing import List

import pytest

import knowledge_base as kb_mod
from knowledge_base import KnowledgeBase

pytestmark = pytest.mark.unit

if not kb_mod.KNOWLEDGE_DEPS_AVAILABLE:  # pragma: no cover
    pytest.skip("langchain-core not installed", allow_module_level=True)

from langchain_core.embeddings import Embeddings  # noqa: E402

_DIM = 1024


class FakeEmbeddings(Embeddings):
    """Deterministic bag-of-words hashing embedder (md5, not salted hash()).

    High dimension keeps token-hash collisions from distorting ranking.
    """

    def __init__(self):
        self.embed_calls = 0

    def _vec(self, text: str) -> List[float]:
        vec = [0.0] * _DIM
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % _DIM
            vec[idx] += 1.0
        return vec

    def embed_documents(self, texts):
        self.embed_calls += len(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def knowledge_dir(tmp_path):
    d = tmp_path / "knowledge"
    d.mkdir()
    (d / "returns.md").write_text(
        "Refunds are issued within fourteen days of purchase. "
        "Contact support with the order number to start a refund.")
    (d / "hours.txt").write_text(
        "The office is open weekdays from nine to five. "
        "Weekend calls are handled by the answering service.")
    return d


@pytest.fixture
def kb(config_factory, tmp_path, knowledge_dir):
    cfg = config_factory(data_dir=str(tmp_path), knowledge_dir=str(knowledge_dir))
    return KnowledgeBase(cfg, embeddings=FakeEmbeddings())


async def test_index_and_search(kb):
    assert kb.available
    assert not kb.ready
    await kb.start()
    assert kb.ready

    results = await kb.search("refund order purchase")
    assert results
    source, chunk = results[0]
    assert source == "returns.md"
    assert "refund" in chunk.lower()


async def test_search_before_index_is_empty(kb):
    assert await kb.search("refund") == []


async def test_format_for_prompt(kb):
    await kb.start()
    block = await kb.format_for_prompt("office hours weekdays")
    assert "[hours.txt]" in block


async def test_persisted_index_skips_reembedding(config_factory, tmp_path,
                                                 knowledge_dir):
    cfg = config_factory(data_dir=str(tmp_path), knowledge_dir=str(knowledge_dir))
    first = KnowledgeBase(cfg, embeddings=FakeEmbeddings())
    await first.start()
    assert first._embeddings.embed_calls > 0

    reloaded = KnowledgeBase(cfg, embeddings=FakeEmbeddings())
    await reloaded.start()
    assert reloaded.ready
    assert reloaded._embeddings.embed_calls == 0  # loaded from cache
    assert await reloaded.search("refund order purchase")


async def test_changed_document_triggers_reindex(config_factory, tmp_path,
                                                 knowledge_dir):
    cfg = config_factory(data_dir=str(tmp_path), knowledge_dir=str(knowledge_dir))
    first = KnowledgeBase(cfg, embeddings=FakeEmbeddings())
    await first.start()

    (knowledge_dir / "returns.md").write_text(
        "Refund policy changed: refunds now take thirty days to process.")
    changed = KnowledgeBase(cfg, embeddings=FakeEmbeddings())
    await changed.start()
    assert changed._embeddings.embed_calls > 0  # re-embedded
    results = await changed.search("refund thirty days")
    assert any("thirty days" in chunk for _, chunk in results)


async def test_empty_dir_is_unavailable(config_factory, tmp_path):
    empty = tmp_path / "knowledge"
    empty.mkdir()
    cfg = config_factory(data_dir=str(tmp_path), knowledge_dir=str(empty))
    kb = KnowledgeBase(cfg, embeddings=FakeEmbeddings())
    assert not kb.available
    await kb.start()  # no-op, no crash
    assert not kb.ready


async def test_disabled_is_unavailable(config_factory, tmp_path, knowledge_dir):
    cfg = config_factory(data_dir=str(tmp_path), knowledge_dir=str(knowledge_dir),
                         knowledge_enabled="false")
    kb = KnowledgeBase(cfg, embeddings=FakeEmbeddings())
    assert not kb.available

async def test_available_is_memoized(kb, monkeypatch):
    """`available` walks the knowledge dir; the KNOWLEDGE tool reads it per
    call on the event loop, so it must be computed once."""
    calls = []
    real = kb._compute_available
    monkeypatch.setattr(kb, "_compute_available",
                        lambda: (calls.append(1), real())[1])
    assert kb.available and kb.available and kb.available
    assert len(calls) == 1
    await kb.start()          # start() re-evaluates once
    assert kb.available
    assert len(calls) == 2


async def test_knowledge_tool_skips_availability_check_when_ready(kb):
    from types import SimpleNamespace
    from plugins.knowledge_tool import KnowledgeTool
    await kb.start()
    tool = KnowledgeTool(SimpleNamespace(config=None, knowledge_base=kb))
    kb._compute_available = lambda: pytest.fail("availability re-checked")
    kb._available = None
    result = await tool.execute({"query": "refund order purchase"})
    assert "returns.md" in result.message
