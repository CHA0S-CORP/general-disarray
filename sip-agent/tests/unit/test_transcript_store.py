"""Unit tests for TranscriptStore.remove_last_turn — the retraction API used
by the speculative cancel-merge path so a cancelled turn's user fragment does
not linger in the persisted transcript next to the merged re-dispatch."""
import pytest

from transcript_store import TranscriptStore

pytestmark = pytest.mark.unit

FRAGMENT = "I need a reservation."


def test_remove_last_turn_retracts_matching_fragment(config):
    store = TranscriptStore(config)
    store.start("call-rm-1", "inbound", "sip:1001@host")
    store.add_turn("call-rm-1", "user", FRAGMENT)

    assert store.remove_last_turn("call-rm-1", "user", FRAGMENT) is True
    assert store.get("call-rm-1")["turns"] == []

    # The merged utterance then lands as the ONLY user turn.
    store.add_turn("call-rm-1", "user", f"{FRAGMENT} for six people tonight.")
    turns = store.get("call-rm-1")["turns"]
    assert [t["content"] for t in turns] == [
        f"{FRAGMENT} for six people tonight."]


def test_remove_last_turn_is_noop_on_mismatch_or_missing(config):
    store = TranscriptStore(config)
    # Unknown call.
    assert store.remove_last_turn("call-rm-none", "user", FRAGMENT) is False
    # Known call, no turns.
    store.start("call-rm-2", "inbound", "sip:1001@host")
    assert store.remove_last_turn("call-rm-2", "user", FRAGMENT) is False
    # Last turn is a different role/content: must not be touched.
    store.add_turn("call-rm-2", "user", FRAGMENT)
    store.add_turn("call-rm-2", "assistant", "Certainly.")
    assert store.remove_last_turn("call-rm-2", "user", FRAGMENT) is False
    assert store.remove_last_turn("call-rm-2", "assistant", "other") is False
    assert [t["content"] for t in store.get("call-rm-2")["turns"]] == [
        FRAGMENT, "Certainly."]
