---
description: "Start a new development task with the full workflow: explore → plan → develop → review → test → validate → document → commit."
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

# New Task Workflow

Follow this workflow step by step for the requested task. Complete each phase before moving to the next.

## Phase 1: Exploration

Use the **Explorer** agent to analyze the codebase:
- Understand the project structure and relevant code
- Map dependencies and existing patterns
- Identify files that will be affected

Summarize findings before proceeding.

## Phase 2: Planning

Use the **Planner** agent to create an implementation plan:
- Break the task into atomic, ordered steps
- Define acceptance criteria for each step
- Identify edge cases that need test coverage
- List potential risks

Present the plan and wait for approval before proceeding.

## Phase 3: Development

Delegate to the **Developer** agent with the approved plan.
Proceed when implementation is complete and `ruff check` passes.

## Phase 4: Code Review

Delegate to the **Reviewer** agent.
If critical issues are found, return to Phase 3 to fix them before proceeding.

## Phase 5: Testing

Delegate to the **Tester** agent.
Proceed only when `uv run pytest` passes with zero failures.

## Phase 6: Validation

Delegate to the **Validator** agent.
Proceed only when the Validator reports **APPROVE**.

## Phase 7: Documentation

Delegate to the **Documenter** agent with a summary of what changed:
- Update `docs/architecture.md` if design decisions or tools changed
- Update `docs/project-structure.md` if directories or conventions changed
- Update `README.md` only if high-level features or `make` commands changed
- Regenerate API docs (`make docs`) if public interfaces changed

Proceed when documentation is consistent with the implementation.

## Phase 8: Commit

Use the **Committer** agent to create the commit:
- Review all changes made during the task
- Determine the conventional commit type and scope
- Compose a commit message in `<type>(<scope>): <description>` format
- Stage and commit the changes using `uv run cz commit` or manual `git commit`

The task is complete when the commit is created.
