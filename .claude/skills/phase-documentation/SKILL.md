---
name: phase-documentation
description: Use this skill at the end of every project phase to produce a structured journal entry capturing decisions, results, and reusable report prose. Trigger whenever the user says "phase done", "document this phase", "wrap up phase N", or after completing any milestone defined in PROJECT_PLAN.md. Each entry follows a strict template so journal entries can be assembled directly into the final LaTeX report with minimal rewriting.
---

# Phase Documentation Skill

## When to use this skill

Invoke this skill at the end of every phase defined in `PROJECT_PLAN.md`. Phase boundaries are explicit ("Stop for documentation" markers). Also invoke if the user says: "wrap up", "phase done", "document this", or "log this".

Do not invoke mid-phase. Mid-phase notes go in code comments or scratch files, not journal entries.

## Why this skill exists

Three reasons:

1. **The course rubric requires AI tool usage to be documented**, and journal entries with structured AI-tool notes feed the appendix directly.
2. **The final report is a 6-hour task on a 36-hour deadline.** Journal entries draft report prose phase by phase so the report becomes assembly, not writing from scratch.
3. **Decision provenance.** When a result is surprising or a reviewer asks "why did you choose X", the journal entry has the answer with a timestamp.

## How to use this skill

### Step 1: Read the template

Read `docs/journal/_TEMPLATE.md`. The template is authoritative. Do not invent new sections; do not skip sections.

### Step 2: Gather context

Before filling the template, collect:
- The phase number and name from `PROJECT_PLAN.md`.
- Files created or modified in this phase (use `git status` and `git diff --stat`).
- MLflow runs from this phase if applicable (`mlflow runs list`).
- Figures generated in this phase (list files under `figures/`).
- Any user messages from this phase containing decisions or vetoes.

### Step 3: Fill the template

Create `docs/journal/phase_N_<short_name>.md` with the template structure filled in. Naming convention: `phase_0_setup.md`, `phase_1_data.md`, etc.

Key rules for filling:

- **Decisions section.** For each non-trivial decision in the phase, write: (a) the decision, (b) the alternatives considered, (c) the reason for the choice in one sentence. Brevity matters; this is not a thesis.

- **Results section.** Numbers go in tables, not prose. Surprises and anomalies go in prose. If a result was unexpected, flag it with `**Surprise:**` so the user can scan for it.

- **Report prose drafts.** Write 1 to 3 paragraphs in LaTeX-ready prose that can be pasted into the final report. No em dashes. No bullet points inside this section; use prose. Use `\cite{}` placeholders where citations belong; collect the actual references in a list at the bottom of the journal entry. Match Mohamed's preferred style: concise, direct, technical but readable.

- **AI tool usage.** For every Claude Code interaction in this phase, log: what was generated, how it was verified (read line by line, ran tests, compared to a known result, etc.). This is the rubric requirement.

- **Risk register.** Three slots: what could break the next phase, what could break the final report, what could break submission. Each is one sentence. Leave a slot empty only if there's genuinely no risk; do not pad.

- **Next phase prep.** Three actions max that need to happen before the next phase starts cleanly (e.g. "install package X", "re-read section Y of PROJECT_PLAN").

### Step 4: Update the AI tool log

Append entries to `docs/ai_tool_log.md` mirroring the AI usage section of the journal entry. The journal is per-phase; the tool log is cumulative.

### Step 5: Confirm with the user

After writing the entry, summarize for the user in plain prose:
- One sentence on what got done.
- The top decision made and why.
- The biggest result or surprise.
- The single most important risk for the next phase.

Then ask: "Ready to proceed to Phase N+1?" Wait for explicit approval. Do not auto-advance.

## Output quality bar

A good journal entry is:
- Skimmable in under 90 seconds.
- Detailed enough that a co-author could pick up the phase without asking questions.
- Has at least one paragraph of report-ready LaTeX prose.
- Honest about what failed or was skipped.

A bad journal entry restates the PROJECT_PLAN, has no specific numbers, and reads like a status report. Avoid this.

## Special handling for failed phases

If a phase failed or was partially skipped, still write the journal entry. Use a `**Phase outcome: PARTIAL**` or `**Phase outcome: FAILED**` marker at the top. The "Lessons" section in the template becomes the most important section in this case. The rubric explicitly allows failed experiments and rewards clear explanation of what was learned.

## Final report assembly

When Phase 5 starts, all journal entries get concatenated and reorganized into the LaTeX report. The "Report prose draft" sections from journals 1 through 4 form roughly 60% of the final report body. The Phase 5 journal documents what additional prose was needed and what was cut.
