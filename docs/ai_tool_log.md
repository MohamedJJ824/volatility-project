# AI Tool Usage Log

This log is required by the IE412 rubric. Every interaction with an AI tool that contributed to the submitted work is logged here. The report's appendix is built from this file.

## Format

For each entry:

```
### YYYY-MM-DD HH:MM, Phase N
**Tool:** <name and version>
**Task:** <what was generated or asked>
**Verification:** <how the output was checked>
```

---

## Entries

### 2026-06-23 (kickoff), Phase -1 (planning)
**Tool:** Claude (claude-opus-4-7) via web interface
**Task:** Brainstormed project ideas from the IE412 brief, scoped the multi-timescale volatility project, produced `PROJECT_PLAN.md` and the `phase-documentation` skill.
**Verification:** Reviewed scope against the rubric (problem definition, motivation, method, data, results, discussion). Cross-checked the 36-hour budget against the actual deadline. Plan to verify each implementation phase as it's built.

### 2026-06-24 05:40, Phase 0 (setup)
**Tool:** Claude Code (claude-opus-4-8)
**Task:** Built the repo skeleton, requirements.txt, .gitignore, a repo-local venv, installed all dependencies, initialized local git, and authored src/data.py.
**Verification:** Ran the import success check (`import torch, arch, yfinance, mlflow` -> ok), recorded all key package versions, smoke-tested the MLflow SQLite backend (create/log/delete an experiment), and confirmed via git status that environment and data artifacts are gitignored.

