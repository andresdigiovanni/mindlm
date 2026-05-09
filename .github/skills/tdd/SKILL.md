---
name: test-driven-development
description: Use when implementing any feature or bugfix, before writing implementation code. Enforces RED-GREEN-REFACTOR cycle with pytest.
---

# Test-Driven Development (TDD)

## Overview

Write the test first. Watch it fail. Write minimal code to pass.

**Core principle:** If you didn't watch the test fail, you don't know if it tests the right thing.

## When to Use

**Always:**
- New features
- Bug fixes
- Refactoring
- Behavior changes

**Exceptions (ask the user):**
- Throwaway prototypes
- Generated configuration files

## The Iron Law

```
NO PRODUCTION CODE WITHOUT A FAILING TEST FIRST
```

Write code before the test? Delete it. Start over.

**No exceptions:**
- Don't keep it as "reference"
- Don't "adapt" it while writing tests
- Delete means delete

Implement fresh from tests. Period.

## Red-Green-Refactor

### RED — Write Failing Test

Write one minimal test showing what should happen.

**Good:**
```python
class TestRetryOperation:
    def test_should_retry_until_success_when_operation_fails_twice(self):
        attempts = 0

        def flaky_operation():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ConnectionError("transient failure")
            return "success"

        result = retry_operation(flaky_operation, max_retries=3)

        assert result == "success"
        assert attempts == 3
```
Clear name, tests real behavior, one thing.

**Bad:**
```python
def test_retry():
    mock = MagicMock(side_effect=[ValueError(), ValueError(), "success"])
    retry_operation(mock)
    assert mock.call_count == 3
```
Vague name, tests mock not code.

**Requirements:**
- One behavior per test
- Clear name: `test_should_{expected}_when_{condition}`
- Real code (no mocks unless unavoidable)

### Verify RED — Watch It Fail

**MANDATORY. Never skip.**

```bash
uv run pytest tests/path/test_module.py::TestClass::test_name -v
```

Confirm:
- Test fails (not errors)
- Failure message is expected
- Fails because feature is missing (not typos)

**Test passes immediately?** You're testing existing behavior. Fix the test.

**Test errors?** Fix the error, re-run until it fails correctly.

### GREEN — Minimal Code

Write the simplest code to pass the test.

**Good:**
```python
def retry_operation(fn: Callable[[], T], max_retries: int) -> T:
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception:
            if attempt == max_retries - 1:
                raise
    raise RuntimeError("unreachable")
```
Just enough to pass.

**Bad:** Adding configurable backoff, logging, circuit-breaker pattern, etc. before there's a test requiring it. YAGNI.

Don't add features, refactor other code, or "improve" beyond the test.

### Verify GREEN — Watch It Pass

**MANDATORY.**

```bash
uv run pytest tests/path/test_module.py -v
```

Confirm:
- Test passes
- All other tests still pass
- No unexpected warnings or errors

**Test fails?** Fix code, not test.

**Other tests fail?** Fix now — do not proceed.

### REFACTOR — Clean Up

After green **only**:
- Remove duplication
- Improve names
- Extract helpers

Keep tests green. Don't add behavior.

### Repeat

Next failing test for next feature.

---

## Test Quality

| Quality | Good | Bad |
|---------|------|-----|
| **Minimal** | One thing. "and" in name? Split it. | `test_validates_email_and_domain_and_whitespace` |
| **Clear** | Name describes behavior | `test1`, `test_it_works` |
| **Shows intent** | Demonstrates desired API | Obscures what the code should do |

---

## Why Tests First?

**"Tests after verify it works"** — Tests written after pass immediately. Passing immediately proves nothing: they might test the wrong thing, miss edge cases, or test implementation instead of behavior.

**"I already manually tested it"** — Manual testing is ad-hoc. No record, can't re-run when code changes, easy to forget cases.

**"TDD slows me down"** — TDD is faster than debugging. Finding bugs before commit is much cheaper than finding them after.

---

## Common Rationalizations

| Excuse | Reality |
|--------|---------|
| "Too simple to test" | Simple code breaks. Test takes 30 seconds. |
| "I'll test after" | Tests passing immediately prove nothing. |
| "Already manually tested" | Ad-hoc ≠ systematic. No record, can't re-run. |
| "Deleting X hours is wasteful" | Sunk cost fallacy. Keeping unverified code is technical debt. |
| "TDD is dogmatic" | TDD is pragmatic — finding bugs before commit is always faster. |

---

## Verification Checklist

Before marking work complete:

- [ ] Every new function/method has a test
- [ ] Watched each test fail before implementing
- [ ] Each test failed for the expected reason (feature missing, not typo)
- [ ] Wrote minimal code to pass each test
- [ ] All tests pass: `uv run pytest`
- [ ] Lint and types pass: `make lint`
- [ ] Tests use real code (mocks only for external I/O)
- [ ] Edge cases covered per `instructions/tests.instructions.md`

Can't check all boxes? You skipped TDD. Start over.

---

## When Stuck

| Problem | Solution |
|---------|----------|
| Don't know how to test it | Write the wished-for API. Write the assertion first. |
| Test too complicated | Design too complicated. Simplify the interface. |
| Must mock everything | Code too coupled. Use dependency injection. |
| Test setup is huge | Extract fixtures. Still complex? Simplify design. |

---

## Example: Bug Fix

**Bug:** Empty email accepted

**RED**
```python
def test_should_reject_empty_email_when_submitting_form():
    result = submit_form(email="")
    assert result.error == "Email required"
```

**Verify RED**
```bash
$ uv run pytest tests/test_forms.py::test_should_reject_empty_email_when_submitting_form -v
FAILED: AssertionError: assert None == "Email required"
```

**GREEN**
```python
def submit_form(email: str) -> FormResult:
    if not email.strip():
        return FormResult(error="Email required")
    ...
```

**Verify GREEN**
```bash
$ uv run pytest tests/test_forms.py -v
PASSED
```

**REFACTOR** — extract validation if multiple fields share the pattern.
