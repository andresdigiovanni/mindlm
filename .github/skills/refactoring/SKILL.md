---
name: refactoring
description: "Technical debt analysis and refactoring methodology. Use when analyzing code smells, evaluating technical debt, planning refactoring work, or improving code quality without changing behavior. Covers code smell catalog, complexity metrics, risk assessment, and refactoring patterns."
---

# Refactoring & Technical Debt

> For coding rules, see `instructions/python.instructions.md`. This skill covers **analysis methodology and refactoring patterns**.

## When to Use

- Analyzing technical debt in existing code
- Planning a refactoring before implementation
- Evaluating whether code is complex enough to warrant refactoring
- Improving code quality without changing external behavior

## Procedure

### 1. Identify — Code Smell Catalog

Scan the codebase for these patterns:

| Smell | Symptom | Risk |
|---|---|---|
| **Long function** | >30 lines or >3 levels of nesting | Hard to test, hard to understand |
| **God class** | Class with >7 methods or >3 responsibilities | Tight coupling, change propagation |
| **Feature envy** | Method uses more data from another class than its own | Wrong location for the logic |
| **Primitive obsession** | Repeated raw types instead of domain objects | No validation, scattered logic |
| **Duplicate logic** | Same pattern in 3+ places | Inconsistent changes, missed fixes |
| **Long parameter list** | Function with >4 parameters | Needs a data object or builder |
| **Dead code** | Unreachable code, unused imports, commented-out blocks | Noise, confusion |
| **Shotgun surgery** | One change requires edits in many files | High coupling |
| **Speculative generality** | Abstractions with only one implementation | Over-engineering, maintenance cost |
| **Missing abstraction** | Repeated if/elif chains or type checks | Open/closed principle violation |

### 2. Measure — Complexity Metrics

Evaluate severity using:

| Metric | Tool | Threshold |
|---|---|---|
| **Cyclomatic complexity** | `ruff` (C901 rule) | >10 per function = refactor |
| **Cognitive complexity** | Manual review | >15 per function = refactor |
| **Function length** | Line count | >30 lines = consider splitting |
| **Class size** | Method count | >7 public methods = consider splitting |
| **Import depth** | Import graph | >3 levels deep = consider restructuring |
| **Test coverage** | `make coverage` | <80% = add tests before refactoring |

### 3. Prioritize — Risk Assessment

For each identified issue, score:

| Factor | Low (1) | Medium (2) | High (3) |
|---|---|---|---|
| **Frequency of change** | Rarely touched | Monthly changes | Weekly changes |
| **Bug history** | No bugs | Occasional bugs | Repeat offender |
| **Coupling** | Isolated module | Used by 2-3 modules | Used everywhere |
| **Complexity** | Simple logic | Moderate nesting | Deep nesting + state |

**Priority** = Frequency × Coupling × Complexity. Address scores ≥12 first.

### 4. Plan — Refactoring Patterns

Common safe refactoring moves:

| Pattern | When to use | Technique |
|---|---|---|
| **Extract function** | Long function with identifiable sub-tasks | Move block → new function with descriptive name |
| **Extract class** | God class with multiple responsibilities | Group related methods + data → new class |
| **Introduce parameter object** | Long parameter lists | Create `dataclass` for related params |
| **Replace conditional with polymorphism** | Repeated type-checking if/elif | Abstract base + concrete implementations |
| **Move method** | Feature envy | Move method to the class it uses most |
| **Inline** | Unnecessary indirection (one-use abstractions) | Remove wrapper, use the thing directly |
| **Rename** | Unclear naming | Rename to reveal intent |

### 5. Execute — Safety Rules

- **Tests first**: Never refactor without passing tests. Run `make test` before starting.
- **One refactoring at a time**: Don't mix behavior changes with structural changes.
- **Preserve behavior**: Refactoring must not change external behavior. Tests should pass before AND after.
- **Small commits**: Each refactoring move gets its own commit — makes it easy to revert.
- **Verify continuously**: Run `make lint` and `make test` after each move.
