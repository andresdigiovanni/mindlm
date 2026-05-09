---
name: subagent-driven-development
description: Use when executing implementation plans with independent tasks in the current session
---

# Subagent-Driven Development

Execute plan by dispatching fresh subagent per task, with two-stage review after each: spec compliance review first, then code quality review.

**Why subagents:** You delegate tasks to specialized agents with isolated context. By precisely crafting their instructions and context, you ensure they stay focused and succeed at their task. They should never inherit your session's context or history — you construct exactly what they need. This also preserves your own context for coordination work.

**Core principle:** Fresh subagent per task + two-stage review (spec then quality) = high quality, fast iteration

## When to Use

**Use when:**
- You have an implementation plan with mostly independent tasks
- You want to stay in this session (vs. parallel sessions)
- You want fast iteration with review gates between tasks

**vs. executing-plans (when to use that instead):**
- executing-plans: separate session, human-in-loop between tasks
- subagent-driven-development: same session, automated review gates, faster iteration

## The Process

1. **Read plan, extract all tasks** with full text, note context, create todo list
2. **Per task:**
   - Dispatch implementer subagent (see `implementer-prompt.md`)
   - Answer any questions the implementer has
   - Dispatch spec compliance reviewer (see `spec-reviewer-prompt.md`)
   - Fix any spec gaps, re-review until passing
   - Dispatch code quality reviewer (see `code-quality-reviewer-prompt.md`)
   - Fix any quality issues, re-review until approved
   - Mark task complete
3. **After all tasks:** Dispatch final code reviewer for entire implementation

## Model Selection

Use the least powerful model that can handle each role to conserve cost and increase speed.

**Mechanical implementation tasks** (isolated functions, clear specs, 1-2 files): use a fast, cheap model.

**Integration and judgment tasks** (multi-file coordination, pattern matching, debugging): use a standard model.

**Architecture, design, and review tasks**: use the most capable available model.

**Task complexity signals:**
- Touches 1-2 files with a complete spec → cheap model
- Touches multiple files with integration concerns → standard model
- Requires design judgment or broad codebase understanding → most capable model

## Handling Implementer Status

Implementer subagents report one of four statuses:

**DONE:** Proceed to spec compliance review.

**DONE_WITH_CONCERNS:** Read the concerns before proceeding. If about correctness or scope, address them before review. If they're observations (e.g., "this file is getting large"), note and proceed.

**NEEDS_CONTEXT:** Provide the missing context and re-dispatch.

**BLOCKED:** Assess the blocker:
1. If context problem → provide more context and re-dispatch with same model
2. If needs more reasoning → re-dispatch with more capable model
3. If task too large → break into smaller pieces
4. If plan itself is wrong → escalate to the user

**Never** ignore an escalation or force the same model to retry without changes.

## Prompt Templates

- `./implementer-prompt.md` - Dispatch implementer subagent
- `./spec-reviewer-prompt.md` - Dispatch spec compliance reviewer subagent
- `./code-quality-reviewer-prompt.md` - Dispatch code quality reviewer subagent
