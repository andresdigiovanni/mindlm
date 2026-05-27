from dataclasses import replace

from mindlm.core.config.models import ChunkingConfig
from mindlm.core.models import Result


class ContextResolver:
    """Resolves parent or sentence-window context for retrieved chunks."""

    def __init__(self, config: ChunkingConfig) -> None:
        self._resolve_parents = config.parent_chunk_size is not None
        self._resolve_windows = config.strategy == "sentence_window"

    def resolve(self, results: list[Result]) -> list[Result]:
        """Apply parent or window resolution if configured.

        Args:
            results: Retrieved results

        Returns:
            Results with content replaced by parent_content or window_context if available.
            Returns results unchanged if neither resolution is configured.
        """
        if self._resolve_parents:
            return self._apply_parent_resolution(results)
        if self._resolve_windows:
            return self._apply_window_resolution(results)
        return results

    def _apply_parent_resolution(self, results: list[Result]) -> list[Result]:
        resolved = []
        for r in results:
            if "parent_content" in r.payload:
                new_payload = {
                    **r.payload,
                    "matched_chunk": r.payload["content"],
                    "content": r.payload["parent_content"],
                }
                resolved.append(replace(r, payload=new_payload))
            else:
                resolved.append(r)
        return resolved

    def _apply_window_resolution(self, results: list[Result]) -> list[Result]:
        resolved: list[Result] = []
        for r in results:
            if "window_context" in r.payload:
                new_payload = {
                    **r.payload,
                    "matched_chunk": r.payload["content"],
                    "content": r.payload["window_context"],
                }
                resolved.append(replace(r, payload=new_payload))
            else:
                resolved.append(r)
        return resolved
