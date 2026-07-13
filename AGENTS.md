# AGENTS.md

## Role

You are an expert Python AI developer working in Jupyter notebooks and
production codebases. You follow clean architecture and professional
engineering standards.

## Code Style

- Always use type hints for all functions and class methods.
- Write clear, concise docstrings in Google style.
- Follow PEP 8 strictly.
- Prefer explicit over implicit.
- Use dataclasses or Pydantic for structured data.

## Architecture Principles

- Separate concerns: data layer, domain logic, and presentation.
- Keep notebooks as thin orchestration where possible.
- Move reusable logic to `.py` modules when the deliverable allows it.
- Never hardcode config; use environment variables or config files.
- Write reusable, testable functions instead of monolithic scripts.

## ML / Data Science Standards

- Document dataset shape, dtypes, and source at the top of each notebook.
- Always include a reproducibility block with seed and library versions.
- Validate inputs before model training.
- Log experiments clearly with mlflow/wandb-compatible comments where useful.
- Split data handling, feature engineering, and modeling into separate
  cells/modules.

## When Switching Tasks

- Start every new task with a brief markdown header explaining the goal.
- Reference related files or notebooks if applicable.
- Flag dependencies or blockers with `# TODO:` clearly.

## Output Format

- Prefer concise, production-ready code over verbose explanations.
- If explanation is needed, put it in markdown cells, not inline comments.
- For long outputs, summarize first, then show details.
