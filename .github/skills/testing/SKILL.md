---
name: testing
description: "Testing workflow with pytest, tox, and edge case methodology. Use when writing tests, running test suites, debugging test failures, or ensuring edge case coverage. Covers test execution, code examples, fixtures, and coverage."
---

# Testing Workflow

> For testing rules and edge case requirements, see `instructions/tests.instructions.md`. This skill covers **procedures and code examples only**.

## When to Use

- Writing new tests for a feature or module
- Running the full test suite
- Debugging failing tests
- Reviewing test coverage

## Procedure

### 1. Run Tests

```bash
make test          # uv run pytest (quick)
make coverage      # pytest + HTML coverage report
make tox           # multi-version (py311, py312, py313)
```

For a specific test: `uv run pytest tests/path/test_file.py::TestClass::test_name`

### 2. Test Structure Example

```python
class TestUserServiceCreate:
    """Tests for UserService.create() method."""

    def test_should_create_user_when_valid_input(self, db_session):
        # Arrange
        service = UserService(db_session)
        user_data = UserCreate(name="Alice", email="alice@example.com")

        # Act
        result = service.create(user_data)

        # Assert
        assert result.id is not None
        assert result.name == "Alice"

    def test_should_raise_when_email_is_none(self, db_session):
        # Arrange
        service = UserService(db_session)

        # Act & Assert
        with pytest.raises(ValidationError, match="email"):
            service.create(UserCreate(name="Alice", email=None))

    @pytest.mark.parametrize(
        "invalid_email",
        [
            "",
            "not-an-email",
            "@missing-local",
            "missing-domain@",
            "a" * 255 + "@test.com",
        ],
        ids=["empty", "no-at-sign", "no-local-part", "no-domain", "too-long"],
    )
    def test_should_raise_when_email_is_invalid(self, db_session, invalid_email):
        # Arrange
        service = UserService(db_session)

        # Act & Assert
        with pytest.raises(ValidationError):
            service.create(UserCreate(name="Alice", email=invalid_email))
```

### 3. Edge Case Methodology

For every function or method under test, systematically cover:

| Category | Examples |
|----------|----------|
| **Empty inputs** | `""`, `[]`, `{}`, `set()`, `None`, `0`, `0.0` |
| **Boundary values** | Min/max int, first/last element, off-by-one, length 1 |
| **Invalid types** | Wrong type passed, `None` where not expected |
| **Error conditions** | Exceptions raised, timeouts, missing resources |
| **Special strings** | Unicode, emoji, whitespace-only, very long strings |
| **Concurrency** | Thread safety, race conditions, deadlocks |
| **Large inputs** | Performance with 10K+ items, memory limits |
| **State transitions** | Initial → final, repeated calls, idempotent operations |

### 4. Fixture Patterns

```python
# conftest.py
@pytest.fixture
def db_session():
    """Provide a clean database session per test."""
    session = create_test_session()
    yield session
    session.rollback()
    session.close()

@pytest.fixture
def mock_api_client(mocker):
    """Mock external API client."""
    return mocker.patch("mypackage.services.ApiClient")
```

### 4. Coverage Check

```bash
make coverage
# then open htmlcov/index.html
```

Coverage threshold is 80% (`fail_under = 80` in `pyproject.toml`). Focus on **branch coverage** — ensure both paths of every conditional are tested.
