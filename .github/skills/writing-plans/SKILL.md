---
name: writing-plans
description: Create detailed implementation plans from approved design documents. Use after brainstorming/design is complete to break work into complexity-tagged tasks with progress checklists. Activates when asked to write a plan, create tasks, break down implementation, or when a design.md exists and the user says "ready to plan" or "create the implementation plan."
---

# Writing Plans

A structured process for converting a validated design document into a concrete, actionable `plan.md` with complexity-tagged tasks and progress tracking.

## Prerequisites

An approved design document must exist before writing a plan. Check for:

1. A `design.md` in `docs/YYYY-MM-DD-<feature-name>/` (same directory where
   `plan.md` will be saved).
2. If no design exists, inform the user:
   > "A validated design is needed before writing a plan. Use the brainstorming
   > skill to create one first."

Read the design document to understand the full scope before planning.

## Core Principles

- **Every task must be independently actionable.** Any engineer or agent should be able to pick up a task and execute it without reading the entire plan.
- **Exact file paths always.** Never say "update the config" — say "update `src/{package}/config.py`."
- **Complete code intent in plan.** Not "add validation" — instead "add a Pydantic validator for `UserInput` rejecting emails without @ and names shorter than 2 chars."
- **Verification steps for every task.** What command to run, what output to expect.
- **Complexity tags on every task.** Each task gets `[LOW]`, `[MEDIUM]`, or `[HIGH]` based on the rubric below.

## Complexity Rubric

| Tag | Criteria | Typical Scope |
| --- | --- | --- |
| `[LOW]` | Single file, mechanical change, clear pattern to follow, no ambiguity | Config changes, adding a field, renaming, simple tests |
| `[MEDIUM]` | 2-4 files, requires understanding context, some decision-making | New function with tests, refactoring a module, extending existing pattern |
| `[HIGH]` | 5+ files or cross-cutting, novel pattern, architectural decisions, error-prone | New subsystem, complex refactors, performance optimization, security-sensitive |

When tagging, err toward higher complexity if uncertain. A task that seems
`[LOW]` but touches shared utilities is `[MEDIUM]`.

## Process

### Step 1: Read and Internalize the Design

Read the `design.md` thoroughly. Identify:

- All components mentioned
- All files that will be created or modified
- All external integrations
- All edge cases listed

### Step 2: Map the Work

Before writing tasks, create a mental map:

- What are the natural phases? (e.g., Types → Core Logic → Tests → Integration)
- What can be parallelized?
- What are the dependencies?

### Step 3: Write the Plan

Use the template in [plan-template.md](./templates/plan-template.md). For each
task, include:

- Complexity tag: `[LOW]`, `[MEDIUM]`, or `[HIGH]`
- Clear title
- Affected file(s) with full paths
- What to do (specific, not vague)
- Verification step
- Dependencies (if any)

### Task Ordering Rules

1. **Tests before implementation** (RED-GREEN TDD when applicable).
2. **Foundation before features** — shared types, utilities, config first.
3. **Independent tasks before dependent ones** — mark dependencies explicitly.
4. **Group by phase** — each phase should be a coherent, shippable increment.

### Task Size Target

Each task should take **2-15 minutes** for an agent to complete. Signs a task is too big:

- It touches more than 4 files
- The description is longer than 10 lines
- It has the word "and" connecting two different concerns

### Verification Steps

Every task MUST include a verification step. Acceptable verifications:

- `Run: uv run pytest tests/path/test_module.py::TestClass::test_name` → `Expected: 1 passed`
- `Run: uv run ruff check src/` → `Expected: No errors`
- `Run: uv run mypy src/` → `Expected: Success: no issues found`
- `Verify: File src/{pkg}/models.py exports UserInput dataclass` (for structural checks)

### Progress Tracking Format

Use GitHub-compatible checkboxes. The plan is a living document to be updated as
tasks complete:

```markdown
- [ ] `[MEDIUM]` **Task 1.1: Create user input schema** — `src/{pkg}/models.py`
```

### Step 4: Review and Validate

Before saving, review the plan against:

- [ ] Every component in the design is covered by at least one task
- [ ] No task is ambiguous — could a stranger execute it?
- [ ] Verification steps are runnable commands, not "check that it works"
- [ ] Complexity tags are justified
- [ ] Task ordering respects dependencies
- [ ] Total task count feels right (typically 8-25 for a medium feature)

Present a summary to the user: total tasks, breakdown by complexity, estimated
phases. Get confirmation before saving.

### Step 5: Save and Report

**File path:** `docs/YYYY-MM-DD-<feature-name>/plan.md`

Use the same directory as the design document.

After saving, report:

> **Plan saved to `docs/YYYY-MM-DD-<feature-name>/plan.md`.**
>
> Summary:
> - **{N} tasks** across **{M} phases**
> - Complexity: {X} LOW, {Y} MEDIUM, {Z} HIGH
>
> The plan includes progress checklists. Update tasks as `[x]` when complete.

## Plan Document Header

**Every plan MUST start with this header:**

```markdown
# [Feature Name] Implementation Plan

> **For agentic workers:** Use the `subagent-driven-development` skill (recommended) or `executing-plans` skill to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** [One sentence describing what this builds]

**Architecture:** [2-3 sentences about approach]

**Tech Stack:** [Key technologies/libraries]

---
```

## No Placeholders

Every step must contain the actual content an engineer needs. These are **plan failures** — never write them:
- "TBD", "TODO", "implement later", "fill in details"
- "Add appropriate error handling" / "add validation" / "handle edge cases"
- "Write tests for the above" (without actual test code)
- "Similar to Task N" (repeat the code — the engineer may be reading tasks out of order)
- Steps that describe what to do without showing how (code blocks required for code steps)
- References to types, functions, or methods not defined in any task

## Anti-Patterns

- **Vague tasks:** "Refactor the auth module" → split into specific steps.
- **Missing file paths:** Every task must name exact files.
- **No verification:** If it can't be verified, it can't be shipped.
- **Monolith tasks:** If it's `[HIGH]` and takes >15 min, split it.
- **Copy-pasting the design:** The plan references the design; it doesn't duplicate it.

## Self-Review

After writing the complete plan, look at it with fresh eyes:

**1. Spec coverage:** Skim each section/requirement in the design. Can you point to a task that implements it? List any gaps.

**2. Placeholder scan:** Search your plan for red flags — any of the patterns from the "No Placeholders" section above. Fix them.

**3. Type consistency:** Do the types, method signatures, and property names you used in later tasks match what you defined in earlier tasks?

If you find issues, fix them inline. If you find a design requirement with no task, add the task.

## Execution Handoff

After saving the plan, offer execution choice:

**"Plan complete and saved to `docs/YYYY-MM-DD-<feature-name>/plan.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using `executing-plans`, batch execution with checkpoints

**Which approach?"
