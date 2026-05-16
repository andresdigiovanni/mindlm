from typing import Any
from unittest.mock import MagicMock

import pytest

from mindlm.core.config.models import ContextualRetrievalConfig
from mindlm.core.ingestion.contextualizer import Contextualizer


def _make_contextualizer(llm: MagicMock, template: str | None = None) -> Contextualizer:
    kwargs: dict[str, Any] = {"enabled": True}
    if template:
        kwargs["prompt_template"] = template
    config = ContextualRetrievalConfig(**kwargs)
    return Contextualizer(config, llm)


class TestContextualizer:
    def test_returns_context_prepended_to_chunk(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = "This chunk discusses pricing."
        ctx = _make_contextualizer(llm)

        result = ctx.contextualize("Full document text.", "Chunk text.")

        assert result == "This chunk discusses pricing. Chunk text."

    def test_llm_called_once(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = "Context."
        ctx = _make_contextualizer(llm)

        ctx.contextualize("doc", "chunk")

        llm.chat.assert_called_once()

    def test_prompt_contains_document_and_chunk(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = "context"
        ctx = _make_contextualizer(llm)

        ctx.contextualize("MY_DOCUMENT", "MY_CHUNK")

        call_args = llm.chat.call_args[0][0]
        prompt = call_args[0]["content"]
        assert "MY_DOCUMENT" in prompt
        assert "MY_CHUNK" in prompt

    def test_empty_llm_response_returns_original_chunk(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = ""
        ctx = _make_contextualizer(llm)

        result = ctx.contextualize("doc", "original chunk")

        assert result == "original chunk"

    def test_whitespace_only_response_returns_original(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = "   "
        ctx = _make_contextualizer(llm)

        result = ctx.contextualize("doc", "original chunk")

        assert result == "original chunk"

    def test_custom_prompt_template_used(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = "ctx"
        ctx = _make_contextualizer(llm, template="DOC={document} CHUNK={chunk}")

        ctx.contextualize("mydoc", "mychunk")

        prompt = llm.chat.call_args[0][0][0]["content"]
        assert prompt == "DOC=mydoc CHUNK=mychunk"

    def test_llm_exception_propagates(self) -> None:
        llm = MagicMock()
        llm.chat.side_effect = RuntimeError("LLM down")
        ctx = _make_contextualizer(llm)

        with pytest.raises(RuntimeError):
            ctx.contextualize("doc", "chunk")
