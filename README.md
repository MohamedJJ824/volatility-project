# Volatility Project (working repo)

Bootstrap layout for the IE412 term project. Open in VSCode, point Claude Code at this directory, and start the agent with the prompt below.

## Kickoff prompt for Claude Code

```
Read PROJECT_PLAN.md fully. Confirm you understand:
1. The north star, deadline, and operating principles.
2. That you must stop after every phase and invoke the phase-documentation skill.
3. That you must stop at every DECISION POINT and ask me before proceeding.

Once confirmed, begin Phase 0.
```

## What's in here

- `PROJECT_PLAN.md`: the spec. Agent reads it every session.
- `.claude/skills/phase-documentation/SKILL.md`: auto-doc skill, fires at every phase boundary.
- `docs/journal/_TEMPLATE.md`: template the skill fills in.
- `docs/ai_tool_log.md`: cumulative AI tool usage log (for the report appendix).

The rest of the structure (src/, data/, etc.) gets created in Phase 0.
