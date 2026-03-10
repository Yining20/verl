# Research Workflow Rules

This repository is used for research experiments. 
All comments must be written in English.
Changes must be minimal, traceable, and reproducible.

## Before modifying code

Always follow this workflow:

1. Analyze the repository structure
2. Identify relevant files
3. Propose a plan
4. Wait for confirmation before editing

The plan must include:

- Problem understanding
- Relevant files
- Proposed modifications
- Expected behavior changes
- Validation method

Do NOT modify code until the plan is approved.

---

## Code modification rules

When implementing changes:

- Modify the smallest number of files possible
- Avoid touching unrelated modules
- Follow existing code style
- Prefer configuration changes over core logic changes
- Keep experiments reproducible

When editing code, always report:

- Files modified
- Exact purpose of each change
- Potential side effects

---

## Running experiments

Before running long experiments:

1. Run a quick validation command
2. Ensure the training pipeline starts correctly
3. Avoid long runs unless explicitly requested

---

## Research logging

After every modification or experiment:

Append a new entry to:

research_logs/agent_notes.md

The entry must contain:

- Experiment goal
- Hypothesis
- Files modified
- Validation command
- Observed results
- Next steps

---

## Safety constraints

Never:

- commit or push code automatically
- modify files outside the repository
- access secrets or environment variables
- refactor large parts of the codebase