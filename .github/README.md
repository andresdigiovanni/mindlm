# GitHub Copilot — Usage Guide

> Human-readable guide. This file is not read by Copilot.

---

## Table of Contents

- [GitHub Copilot — Usage Guide](#github-copilot--usage-guide)
  - [Table of Contents](#table-of-contents)
  - [Quick Start](#quick-start)
  - [How Copilot Reads Context](#how-copilot-reads-context)
  - [Folder Structure](#folder-structure)
  - [Interactive Mode](#interactive-mode)
    - [copilot-instructions.md](#copilot-instructionsmd)
    - [instructions/](#instructions)
    - [skills/](#skills)
    - [agents/](#agents)
    - [prompts/](#prompts)
    - [hooks/](#hooks)
  - [Autonomous Mode](#autonomous-mode)
    - [workflows/](#workflows)
  - [Beyond This Folder](#beyond-this-folder)
    - [MCP — External Tool Servers](#mcp--external-tool-servers)
    - [Content Exclusion](#content-exclusion)
    - [Model Selection](#model-selection)
  - [Quick Reference](#quick-reference)

---

## Quick Start

**Starting a new task:**
```
In VS Code chat, type: /new-task
```
This runs the full workflow: Explorer → Planner → Developer → Reviewer → Tester → Validator → Committer.

**Invoking an agent directly:**
Select it from the agent dropdown in the VS Code Chat panel, or mention it in your prompt.

**Checking what context Copilot is using:**
After each response, expand the **References** section at the bottom. It lists every instruction file that was injected. If a file is missing, it's not being applied.

---

## How Copilot Reads Context

Copilot doesn't just respond to your prompt — it layers multiple context sources before answering. Understanding this model is the key to configuring it well.

```
Every message →  copilot-instructions.md       always active
                 instructions/ (applyTo)        active when file type matches
                 skills/                        active when task is relevant
                 agents/                        active when agent is selected
                         ↓
                 Your prompt
                         ↓
                 Response
```

Two operating modes exist:

| Mode | You are present | Triggered by |
|---|---|---|
| **Interactive** | Yes | Your prompts in VS Code / CLI |
| **Autonomous** | No | GitHub events, schedules (GitHub Actions) |

Interactive uses `copilot-instructions.md`, `instructions/`, `skills/`, `agents/`, `prompts/`, and `hooks/`.
Autonomous uses `workflows/`.

---

## Folder Structure

```
.github/
├── copilot-instructions.md   ← global rules, always active
├── instructions/             ← rules by domain/file type (auto-applied)
├── skills/                   ← specialized knowledge, loaded on demand
├── agents/                   ← role-based agents with defined tools
├── prompts/                  ← reusable slash commands (/name)
├── hooks/                    ← scripts on session events (lint, guardrails)
└── workflows/                ← autonomous automation via GitHub Actions
```

**Configured outside this folder:**
- `.vscode/mcp.json` — MCP servers (external tool integrations)
- GitHub.com → repo Settings → Copilot → Content exclusion

---

## Interactive Mode

These elements are active during a live session with Copilot.

---

### copilot-instructions.md

The base context Copilot receives in **every single conversation**, automatically.

- **Activate:** Always, no action needed
- **Put here:** Stack, architecture, core conventions, key constraints (keep < 25 lines)
- **Don't put here:** Domain-specific rules or detailed patterns — those belong in `instructions/` or `skills/`

> Everything in this file costs context on every message. Keep it concise.

---

### instructions/

Rules that activate **automatically based on the file you are editing**. Requires an `applyTo` glob in the frontmatter — without it, the file is ignored.

**Files in this repo:**

| File | Activates on |
|---|---|
| `python.instructions.md` | `**/*.py` |
| `tests.instructions.md` | `**/test_*.py`, `**/tests/**/*.py` |

**Adding a new instruction:**
```markdown
---
applyTo: "**/*.ts"
---

Use strict TypeScript. Prefer `interface` over `type` for object shapes.
```

---

### skills/

Packaged specialized knowledge Copilot loads **only when the task is relevant**. Unlike instructions, skills don't activate based on the file — they activate based on what you're asking.

The `description` field in `SKILL.md` is what Copilot reads to decide whether to load a skill. Write it as an explicit trigger: `"Use when..."`.

**Skills in this repo:**

| Skill | When it activates |
|---|---|
| `python-dev/` | Implementing or refactoring Python code |
| `testing/` | Writing tests, running pytest/tox, covering edge cases |
| `refactoring/` | Analyzing technical debt, planning refactoring, reducing complexity |

**Adding a new skill:**
```
skills/
└── my-skill/
    ├── SKILL.md        ← required, exact filename
    └── examples.py     ← optional: real code examples from this repo
```

---

### agents/

Specialized versions of Copilot with a **defined role, specific tools, and constraints**. Each agent is an expert in one phase of the development cycle.

Use agents when you want to delegate a full task to a role, not just ask a question.

**Available agents:**

| Agent | Role | When to use |
|---|---|---|
| `Explorer` | Code exploration (read-only) | Understand existing code, map dependencies |
| `Planner` | Planning (read-only) | Design an implementation, break down a task |
| `Developer` | Implementation | Write code, fix bugs, refactor |
| `Reviewer` | Code review (read-only) | Review changes, validate standards, audit security |
| `Tester` | Testing | Write tests, run pytest/tox, verify coverage |
| `Validator` | Final validation (read-only) | Verify everything before committing |
| `Documenter` | Documentation | Update `docs/` and `README.md` after implementation |
| `Committer` | Commit | Compose and execute the commit message |

**Recommended workflow:**
```
Explorer → Planner → Developer → Reviewer → Tester → Validator → Documenter → Committer
```
Or launch it all at once with `/new-task`.

---

### prompts/

Reusable prompt templates invoked as **slash commands** in the VS Code chat.

**How to use:** Type `/` in the chat and select from the list.

**Available prompts:**

| Prompt | Description |
|---|---|
| `new-task` | Full development workflow: explore → plan → develop → review → test → validate → document → commit |
| `refactor` | Technical debt analysis and refactoring: explore → plan → refactor → review → test → validate → document → commit |

Use a prompt when the task follows a repeatable structure. Talk directly to Copilot for one-off or exploratory work.

---

### hooks/

Scripts that run **automatically on session lifecycle events** — no action needed from you. Defined as JSON files in this folder.

**How they work:** Each hook declares an event and a shell command. On `preToolUse`, returning exit code `1` blocks the operation; exit `0` approves it.

**Hooks in this repo:**

| File | Event | What it does |
|---|---|---|
| `post-edit.json` | `postToolUse` | Runs `ruff check` after each Python file edit |

**Available events:**

| Event | When it fires |
|---|---|
| `sessionStart` | Session begins |
| `preToolUse` | Before Copilot uses any tool — can approve or deny |
| `postToolUse` | After Copilot uses a tool — can scan output |
| `sessionEnd` | Session ends |

Use hooks for policies that must hold deterministically: security guardrails, lint checks, audit logging.

---

## Autonomous Mode

These elements run **without you being present**, triggered by GitHub events.

---

### workflows/

Markdown files that compile to **GitHub Actions workflows** and run an AI agent automatically.

> **Not yet configured for this repository.** The section below documents the pattern for future use.

Use for recurring tasks that don't need human involvement: issue triage, weekly reports, post-merge changelogs.

**How to create one:**
1. Write a `.md` file with YAML frontmatter (`on`, `permissions`, `safe-outputs`) and natural language instructions in the body
2. Compile: `gh aw compile .github/workflows/my-workflow.md`
3. Commit the generated YAML — GitHub runs it on the configured trigger

**Hooks vs. Workflows:**

| | Hooks | Workflows |
|---|---|---|
| Runs during | Active Copilot session | GitHub Actions (no session) |
| Triggered by | Session events | GitHub events, schedules |
| Requires you | Yes | No |

**Example:**
```markdown
---
on:
  issues:
    types: [opened]
permissions:
  issues: read
safe-outputs:
  comment-on-issue: true
  add-labels: true
engine: copilot
---

Analyze the new issue. Add appropriate labels and post a comment
acknowledging it. Ask for missing info if the report is incomplete.
```

```bash
gh extension install github/gh-aw
gh aw compile .github/workflows/issue-triage.md
```

> Start with read-only or comment-only `safe-outputs`. Enable write operations only after validating output quality.

---

## Beyond This Folder

Configuration that lives outside `.github/` but is relevant to how Copilot behaves.

---

### MCP — External Tool Servers

MCP servers give Copilot access to external services: GitHub issues/PRs, databases, browser automation, APIs.

**Where to configure:**
- **VS Code workspace:** `.vscode/mcp.json`
- **VS Code user (global):** VS Code Settings
- **Per-agent:** `mcp-servers` property in the agent's `.md` frontmatter

**Example `.vscode/mcp.json`:**
```json
{
  "servers": {
    "github": {
      "type": "http",
      "url": "https://api.githubcopilot.com/mcp"
    },
    "playwright": {
      "command": "npx",
      "args": ["-y", "@microsoft/mcp-server-playwright"]
    }
  }
}
```

> **Security:** Once configured for the cloud agent, Copilot uses MCP tools autonomously without asking. Review permissions carefully before adding a server.

---

### Content Exclusion

Prevents Copilot from reading specific files. Configure at GitHub.com → repo → **Settings → Copilot → Content exclusion**.

**What to exclude:**
```
**/.env, **/.env.*              ← credentials
**/legacy/**, **/deprecated/**  ← code not to use as reference
**/dist/**, **/build/**         ← generated files
**/fixtures/**, **/testdata/**  ← test data
**/*.pem, **/*.key              ← private keys
```

**Important:** Content Exclusion only applies to inline completions and Ask mode. It does **not** apply to Agent/Edit mode or the CLI. For those, add explicit rules in `copilot-instructions.md`:
```markdown
Never read or modify files in `legacy/`, `deprecated/`, `dist/`, or `migrations/`.
Never read `.env` files or any file containing credentials.
```

---

### Model Selection

Copilot supports multiple models. The right choice has a real impact on quality.

| Category | Examples | Best for |
|---|---|---|
| Fast | GPT-5 mini, Claude Haiku 4.5 | Boilerplate, docs, autocomplete |
| General | GPT-4.1, Claude Sonnet 4 | Most tasks — default via **Auto** |
| Deep reasoning | o3, Claude Opus 4, Claude Sonnet 4 Thinking | Architecture, complex refactors, hard bugs |

**Auto mode** (recommended default): Copilot picks the best available model. Also gives a 10% discount on premium requests.

**Thinking Effort** — adjusts reasoning depth without changing model or cost (VS Code model picker):
```
Low    → boilerplate, simple snippets
Medium → regular features, debugging  (default)
High   → complex refactors, hard bugs
x-High → architecture, critical decisions
```

**Fixing a model in an agent or prompt:**
```markdown
---
model: claude-opus-4    ← deep reasoning for architecture agent
---
```

---

## Quick Reference

| Need | Use |
|---|---|
| Copilot always follows a rule | `copilot-instructions.md` |
| Rule only for `.py` files | `instructions/` with `applyTo: "**/*.py"` |
| Project-specific patterns and examples | `skills/` |
| Delegate a full task to a role | `agents/` |
| Reusable workflow with one command | `prompts/` (`/name`) |
| Block dangerous commands automatically | `hooks/` (`preToolUse`) |
| Lint after every edit automatically | `hooks/` (`postToolUse`) |
| Triage issues without being present | `workflows/` |
| Copilot accesses GitHub / a database | MCP server (`.vscode/mcp.json`) |
| Keep credentials out of Copilot's context | Content Exclusion (GitHub Settings) |
| Hard architecture or debugging problem | Deep reasoning model + Thinking Effort High |
