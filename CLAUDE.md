# LLMOps Course on Databricks — Project Guidelines

## Goal and Background

This prepository is a show case of LLMops practices on databricks in context of an organized course.
The repository of the teachers is availabe in the ../course-code-hub/

We will largely follow the guidance of the teachers repository which gets updated regularly with the new code examples from the completed lessons.
We will use a slightly different use case that still needs to be determined, i.e. we will liely use different data than the arxiv data if we find a suitable source.
Until then, we can follow that code pretty much verbatim. Since this is an educative project, most prompts will strive towards understanding and learnign of the owner of the current repository.

## Development Environment

This project uses `uv` for dependency management and running tools.
Python **3.12** is required (matches Databricks Serverless Environment 4).
### Running Commands

**ALWAYS use `uv run` prefix for all Python tools:**

```bash
# Linting, formatting
uv run prek run --all-files

# Running tests
uv run pytest
```

## Project Structure

```
llmops-databricks-course-datajanko/
├── .claude/
│   └── commands/           # Claude Code slash commands (fix-deps, run-notebook, ship)
├── .github/
│   └── workflows/ci.yml
├── notebooks/              # Databricks-format notebooks
├── resources/              # Databricks Asset Bundle job definitions (*.yml)
├── tests/
├── databricks.yml          # Databricks Asset Bundle configuration
├── pyproject.toml
└── version.txt
```

## Dependency Management

### Pinning Rules

**Regular dependencies** (`[project] dependencies`): pin to exact version.
```toml
"pydantic==2.11.7"
"databricks-sdk==0.85.0"
```

**Optional / dev dependencies**: use `>=X.Y.Z,<NEXT_MAJOR`.
```toml
"pytest>=8.3.4,<9"
```

### Packages That Must Always Be Optional

Never put these in `[project] dependencies`:
- `databricks-connect` → `dev` extra
- `ipykernel` → `dev` extra
- `pytest`, `pre-commit` → `ci` extra

### Updating Dependencies

Use the `/fix-deps` skill to look up the latest PyPI versions and update `pyproject.toml` automatically.

After any dependency changes, validate the environment resolves:
```bash
uv sync --extra dev
```

## Skills

Custom slash commands are defined in `.claude/commands/`. Use them to automate common workflows:

| Skill | Command | Description |
|-------|---------|-------------|
| Fix dependencies | `/fix-deps` | Look up latest PyPI versions and update `pyproject.toml` |
| Run notebook | `/run-notebook <path>` | Deploy and run a notebook on Databricks via Asset Bundles |
| Ship | `/ship` | Commit all changes with a structured message and push (blocks on `main`) |

### `/run-notebook`

Deploys the project wheel and runs a notebook as a Databricks job.

```bash
/run-notebook notebooks/hello_world.py
```

What it does:
1. Derives a job resource key from the notebook filename (e.g. `hello_world_job`)
2. Ensures `resources/` exists and is included in `databricks.yml`
3. Creates `resources/<key>.yml` if it doesn't exist, with `env`, `git_sha`, and `run_id` base parameters
4. Runs `databricks bundle deploy` then `databricks bundle run <key>`

## Notebook File Format

All Python files in `notebooks/` must be formatted as Databricks notebooks:

- **First line**: `# Databricks notebook source`
- **Cell separator**: `# COMMAND ----------` between logical sections

This enables running them interactively in both VS Code (via the Jupyter extension) and Databricks.

```python
# Databricks notebook source
"""
Example description.
"""

import os

# COMMAND ----------

print("Hello, world!")
```

**NEVER** use `#!/usr/bin/env python` shebangs in notebook files.
