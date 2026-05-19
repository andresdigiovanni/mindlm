from typing import Any
from unittest.mock import MagicMock

import pytest

from mindlm.core.config.models import ContextualRetrievalConfig
from mindlm.core.ingestion.contextualizer import Contextualizer


def _make_contextualizer(llm: MagicMock, template: str | None = None) -> Contextualizer:
    kwargs: dict[str, Any] = {
        "chunk_context_enabled": True,
        "document_summary_enabled": True,
    }
    if template:
        kwargs["prompt_template"] = template
    config = ContextualRetrievalConfig(**kwargs)
    return Contextualizer(config, llm)


class TestContextualizer:
    def test_returns_context_string(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = "This chunk discusses pricing."
        ctx = _make_contextualizer(llm)

        result = ctx.contextualize("Full document text.", "Chunk text.")

        assert result == "This chunk discusses pricing."

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

    def test_empty_llm_response_returns_empty_string(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = ""
        ctx = _make_contextualizer(llm)

        result = ctx.contextualize("doc", "original chunk")

        assert result == ""

    def test_whitespace_response_returns_empty_string(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = "   "
        ctx = _make_contextualizer(llm)

        result = ctx.contextualize("doc", "original chunk")

        assert result == ""

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

    def test_disabled_chunk_context_returns_empty_without_llm_call(self) -> None:
        llm = MagicMock()
        config = ContextualRetrievalConfig(chunk_context_enabled=False)
        ctx = Contextualizer(config, llm)

        result = ctx.contextualize("doc", "chunk")

        assert result == ""
        llm.chat.assert_not_called()


class TestContextualizerSummarize:
    def test_summarize_returns_stripped_llm_output(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = "  A document about finance.  "
        ctx = _make_contextualizer(llm)

        result = ctx.summarize("doc")

        assert result == "A document about finance."

    def test_summarize_calls_llm_once(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = "Summary."
        ctx = _make_contextualizer(llm)

        ctx.summarize("doc")

        llm.chat.assert_called_once()

    def test_summarize_prompt_contains_document(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = "Summary."
        ctx = _make_contextualizer(llm)

        ctx.summarize("MY_DOCUMENT_TEXT")

        prompt = llm.chat.call_args[0][0][0]["content"]
        assert "MY_DOCUMENT_TEXT" in prompt

    def test_summarize_uses_document_summary_prompt_template(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = "Summary."
        config = ContextualRetrievalConfig(
            document_summary_enabled=True,
            document_summary_prompt_template="SUMMARIZE: {document}",
        )
        ctx = Contextualizer(config, llm)

        ctx.summarize("mydoc")

        prompt = llm.chat.call_args[0][0][0]["content"]
        assert prompt == "SUMMARIZE: mydoc"

    def test_summarize_empty_llm_response_returns_empty_string(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = ""
        ctx = _make_contextualizer(llm)

        result = ctx.summarize("doc")

        assert result == ""

    def test_summarize_whitespace_response_returns_empty_string(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = "   "
        ctx = _make_contextualizer(llm)

        result = ctx.summarize("doc")

        assert result == ""

    def test_summarize_llm_exception_propagates(self) -> None:
        llm = MagicMock()
        llm.chat.side_effect = RuntimeError("LLM down")
        ctx = _make_contextualizer(llm)

        with pytest.raises(RuntimeError):
            ctx.summarize("doc")

    def test_disabled_document_summary_returns_empty_without_llm_call(self) -> None:
        llm = MagicMock()
        config = ContextualRetrievalConfig(document_summary_enabled=False)
        ctx = Contextualizer(config, llm)

        result = ctx.summarize("doc")

        assert result == ""
        llm.chat.assert_not_called()
