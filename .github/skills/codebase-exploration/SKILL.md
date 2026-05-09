---
name: codebase-exploration
description: Systematic codebase reconnaissance and mapping. Use when asked to explore, understand, map, or survey a codebase or a specific area of code before planning, refactoring, or implementing changes. Activates for prompts like "explore the codebase", "understand how X works", "map the architecture", "what does this area look like", or any pre-planning reconnaissance.
---

# Codebase Exploration

A structured process for quickly mapping a codebase or a specific area within it.
Produces a compressed, actionable summary — not a tour of every file.

## Guiding Principles

- **Breadth first, depth on example-demand.** Start wide, narrow only to what's relevant.
- **Facts only.** Report structure, patterns, and dependencies. No recommendations.
- **Compress aggressively.** The output feeds into a planning context with limited
  budget. Every line must earn its place.
- **Respect scope.** If asked about "the auth module," don't map the entire repo.

## Exploration Steps

### 0. Project Type Detection

Identify the project layout:

- Standard Python project: `src/{package_name}/` with `pyproject.toml` and `hatchling` build
- Check for entry points: `main.py`, `cli.py`, `server.py`, `pipeline.py`

### 1. Structural Scan

- List top-level directories to understand project organization.
- Focus exploration on `src/` and `tests/` directories.
- Identify project type: language, framework, build system, package manager.
- Locate entry points.
- Find configuration: `pyproject.toml`, `Makefile`, `tox.ini`, `.github/`, etc.

### 2. Targeted Search

Based on the topic/area requested:

- **File search** — find files by name patterns related to the topic.
- **Text search** — find symbols, function names, class names, constants.
- **Usage search** — for key symbols, find all callers/consumers.

Prioritize:
- Public interfaces and exported symbols over internal helpers.
- Files with many dependents over leaf files.
- Recently modified files (if visible from git) over stale ones.

### 3. Pattern Recognition

In the relevant files, identify:

- **Architecture style:** layered, hexagonal, MVC, microservices, monolith.
- **Naming conventions:** casing, prefixes, suffixes for files/classes/functions.
- **Error handling:** custom exceptions, result types, try/catch patterns.
- **Data flow:** how data enters, transforms, and exits the relevant area.
- **Testing patterns:** framework used, file naming, fixture patterns, mocking approach.

### 4. Dependency Mapping

For the key files in the relevant area:

- What do they import? (internal and external dependencies)
- What imports them? (fan-out / fan-in)
- Are there shared utilities, base classes, or common interfaces?
- Are there circular or surprising dependencies?

### 5. Risk Flagging

Note anything that a planner or implementer should know:

- Files with high complexity or many responsibilities.
- `TODO`, `FIXME`, `HACK` comments in the area.
- Linter warnings or type errors (from diagnostics if available).
- Files with no test coverage.
- Tightly coupled modules that would make changes risky.

## Output Structure

Return findings using these sections. Omit any section that has nothing
relevant to report. Keep total output under 300 lines.

```
## Codebase Exploration: {topic}

### Project Structure
### Key Files
### Patterns & Conventions
### Dependencies & Coupling
### Existing Tests
### Risks & Notes
```

Each "Key Files" entry: one line, format `path/to/file.py` — 1-sentence purpose.

Each "Risks & Notes" entry: one line, actionable. Not "this is complex" but
"auth_manager.py has 14 dependents — changes here cascade widely."
