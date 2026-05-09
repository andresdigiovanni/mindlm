class LLMUnavailableError(RuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class CollectionNotFoundError(RuntimeError):
    def __init__(self, collection: str) -> None:
        super().__init__(f"Collection not found: {collection}")
        self.collection = collection


class ParseError(RuntimeError):
    def __init__(self, path: str, strategy: str) -> None:
        super().__init__(f"Cannot parse {path!r} with strategy {strategy!r}")
        self.path = path
        self.strategy = strategy


class EmbeddingError(RuntimeError):
    def __init__(self, model: str) -> None:
        super().__init__(f"Embedding failed for model {model!r}")
        self.model = model
