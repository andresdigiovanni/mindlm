---
description: "Analyze technical debt and refactor code. Explores the codebase for code smells, plans safe refactoring moves, implements them, and validates behavior is preserved."
agent: agent
tools:
  - agent/runSubagent
  - edit/createDirectory
  - edit/createFile
  - edit/editFiles
  - execute/runInTerminal
  - execute/getTerminalOutput
  - execute/testFailure
  - read/readFile
  - read/problems
  - read/terminalLastCommand
  - search/codebase
  - search/fileSearch
  - search/listDirectory
  - search/textSearch
  - search/usages
  - search/changes
  - vscode/askQuestions
---

# Refactoring Workflow

Follow this workflow to analyze technical debt and refactor safely. Use the **refactoring** skill for methodology.

## Phase 1: Analysis

Use the **Explorer** agent to analyze the codebase for technical debt:
- Read `docs/` for existing architecture context
- Identify code smells using the catalog in the **refactoring** skill
- Measure complexity (function length, nesting depth, cyclomatic complexity)
- Map coupling between modules

Present a **Technical Debt Report**:
```
## Technical Debt Report

### High Priority (score ≥12)
- [file:function] smell — impact — suggested refactoring

### Medium Priority (score 6-11)
- [file:function] smell — impact — suggested refactoring

### Low Priority (score <6)
- [file:function] smell — impact
```

Wait for approval on which items to address before proceeding.

## Phase 2: Planning

Use the **Planner** agent to create a refactoring plan:
- Order refactoring moves from least risky to most risky
- Each move must be independently committable
- Define how to verify behavior is preserved (which tests to run)

Present the plan and wait for approval before proceeding.

## Phase 3: Refactoring

Delegate to the **Developer** agent with the approved plan:
- Run `make test` before starting — all tests must pass
- Implement **one refactoring move at a time**
- Run `make lint` and `make test` after each move
- Do NOT mix behavior changes with structural changes

## Phase 4: Code Review

Delegate to the **Reviewer** agent:
- Verify refactoring did not change external behavior
- Check that code is cleaner, not just different
- If issues found, return to Phase 3

## Phase 5: Testing

Delegate to the **Tester** agent:
- Verify all existing tests still pass
- Add tests if refactoring exposed untested paths
- Proceed only when `make test` passes with zero failures

## Phase 6: Validation

Delegate to the **Validator** agent.
Proceed only when the Validator reports **APPROVE**.

## Phase 7: Documentation

Delegate to the **Documenter** agent:
- Update `docs/architecture.md` if structural patterns changed
- Update `docs/project-structure.md` if directories were added/removed/renamed

## Phase 8: Commit

Use the **Committer** agent to create the commit:
- Use type `refactor` for structural changes: `refactor(scope): description`
- One commit per independent refactoring move when possible
