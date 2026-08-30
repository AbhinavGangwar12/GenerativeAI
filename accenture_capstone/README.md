# 🩺 HealthBot — Patient Health Education Assistant

A LangGraph workflow (Jupyter notebook) that lets a patient pick a health topic, reads a
Tavily-sourced, patient-friendly summary, takes a one-question comprehension check, and
gets a graded, citation-backed explanation — then loops to a new topic or exits.

## Project Structure

```
main.ipynb        # the LangGraph workflow (this is the graded deliverable)
requirements.txt  # dependencies
config.env         # API keys (not committed — see Setup)
README.md         # this file
```

## Setup

1. `pip install uv`
2. `uv init`
3. `uv venv --python 3.11.13`
4. Activate the venv (Windows PowerShell: `.venv\Scripts\Activate`)
5. Create `config.env` in the project folder:
   ```
   OPENAI_API_KEY="sk-***********"
   TAVILY_API_KEY="tvly-***********"
   ```
6. `uv add -r requirements.txt`
7. Open `main.ipynb` and run all cells. The last cell starts an interactive session —
   `input()` prompts appear as notebook modal boxes, output prints below each cell.

## Design Decisions and Why

### Jupyter notebook interface, not Streamlit
The project instructions explicitly require patient input via **Jupyter notebook's
`input()` function** and output via **`print()`**, and require the notebook itself as the
submission. A Streamlit app would run as a separate server process outside that
environment. Instead, effort went into making the `input()`/`print()` experience itself
readable — banners, section dividers, and emoji markers around the summary, question, and
grade output (see the "Display helpers" section of the notebook) — while staying inside
the required interface.

### No `interrupt()` / checkpointer for human-in-the-loop
LangGraph's `interrupt()` exists to pause graph execution **across process boundaries** —
e.g. a webhook handler returns while waiting on an external event (a Slack reply, a GitHub
comment) and resumes later, potentially in a different process, via `Command(resume=...)`
against a checkpoint. HealthBot's HITL steps (topic, ready-check, quiz answer, continue)
all happen synchronously, inline, in a single `graph.invoke()` call within one running
Python process. Python's built-in `input()` already blocks execution exactly where needed
— there's no separate resume step, so nothing needs to be checkpointed or restored.
Reaching for `interrupt()` + `InMemorySaver`/Postgres here would add real complexity to
solve a problem (cross-process resumability) this notebook doesn't have.

### No dedicated short-term memory (STM) layer
Within a single topic, LangGraph's own `State` object already carries everything the flow
needs across nodes — the search results, the summary, the question, the answer. A
dedicated STM layer (a conversation buffer, sliding window, or message-history summarizer)
exists to manage context-window pressure across long, open-ended multi-turn dialogue. Here
each state field is small (a topic name, a handful of search snippets, a 3–4 paragraph
summary) and the full context comfortably fits in a single prompt — there's no long
conversation to compress. The typed `State` dict is a complete, sufficient STM for this
workflow.

### No long-term memory (LTM) across sessions
The project instructions explicitly require state to reset between topics "to maintain
privacy and accuracy," and by extension we don't persist any patient data — topics
researched, answers given, grades received — beyond the running session. An LTM layer
(vector store, user profile database) would mean storing potentially sensitive
health-education interactions per patient across sessions, which raises real
privacy/consent questions this prototype has no authentication, encryption, or compliance
story to support. The design is intentionally session-scoped and ephemeral.

### No RAG / chunking for grading citations
We considered chunking the summary (or raw search results) for retrieval-based citations,
and separately considered scoring the patient's answer via chunk-level precision/recall.
Both were dropped:
- The summary the grader cites from is only a few hundred tokens — well under the context
  window — so retrieval solves a context-size problem that doesn't exist at this scale.
- Lexical-overlap scoring (precision/recall) penalizes a patient who answers correctly in
  their own words, which is the common case, and doesn't produce the required written
  justification on its own.

Instead, the full summary is passed directly into the grading prompt, the model is
required to quote it verbatim, and that quote is verified programmatically after the fact
(`verify_citation()` in the notebook) — with one automatic retry and a real fallback
citation (nearest-matching sentence via `difflib`) if the model still doesn't comply. This
gives reliable, grounded citations without a retrieval pipeline.

### Model: Ollama in development, OpenAI for submission
`ChatOllama` (local, free) was used while iterating on prompts and graph structure, to
avoid burning API credits during development. The submission swaps to `ChatOpenAI`, per
the project's environment setup instructions — see the commented-out import at the top of
the notebook.

## Rubric Coverage

| Rubric criterion | Where it's satisfied |
|---|---|
| API keys loaded correctly | Setup cell — `load_dotenv('config.env')` + asserts |
| Tavily search called for the topic | `search_tavily_node` |
| Summarization — 3–4 paragraphs, summary-only source | `summary_node` prompt |
| Quiz question answerable from summary alone | `generate_question_node` prompt |
| Grading with citation from summary | `grade_node` + `verify_citation()` fallback |
| State object referenced/updated by nodes | `State` TypedDict; every node returns a partial update |
| Nodes with single responsibility | separate ask / present / generate / grade nodes throughout |
| Conditional edges for restart/exit, with state reset | `route_ready`, `route_continue`, `reset_node` |
| Full workflow executes end-to-end | final `graph.invoke()` cell |

## Known Limitations / Future Work

- One quiz question per topic (could extend to a multi-question set).
- No persistent patient profile or history — deliberate, see privacy note above.
- CLI-style interface only; a hosted/web version would need a different I/O layer
  (and, if deployed as a multi-process service, would be exactly the case where
  `interrupt()` + a real checkpointer becomes the right tool).