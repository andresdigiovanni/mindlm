from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import mindlm.api.dependencies as deps
from mindlm.api.main import app


@pytest.fixture(autouse=True)
def clear_overrides() -> Generator[None, None, None]:
    yield
    app.dependency_overrides.clear()


class TestCollectionsEndpoint:
    def test_list_collections(self) -> None:
        mock_vs = MagicMock()
        mock_vs.list_collections.return_value = ["col_a", "col_b"]
        app.dependency_overrides[deps.get_vectorstore] = lambda: mock_vs

        client = TestClient(app)
        response = client.get("/collections")

        assert response.status_code == 200
        assert response.json() == ["col_a", "col_b"]
