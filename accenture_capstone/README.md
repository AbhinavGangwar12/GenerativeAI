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

### Search: model-initiated tool calling, not a direct SDK call
An earlier version called the raw `tavily-python` SDK directly from a Python function.
That works, but doesn't match the rubric's literal wording — "Use the Tavily search
engine (via LangChain community tool)" and "OpenAI successfully calls Tavily for search"
describe the **model** deciding to invoke the tool, not the code calling Tavily and handing
the LLM a result afterward. The current version binds `TavilySearchResults` (the LangChain
community tool) to the LLM with `.bind_tools()`; `search_agent_node` lets the model issue
the tool call itself, and LangGraph's prebuilt `ToolNode` executes it. `force_search_node`
is a safety net for the rare case the model doesn't call the tool on its own, so the
workflow still completes. This also gives a real message trace — `state["messages"]`
carries the tool call, the tool result, the summary, the quiz question, and the grade — so
"the model has access to previous messages (tool calls, summary, quiz question, etc)" is
satisfied by an actual message history rather than separate untracked fields. `reset_node`
clears that history with `RemoveMessage` between topics, same privacy reasoning as the rest
of the state.

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
| Tavily called via the LangChain community tool | `TavilySearchResults` bound to the LLM in `search_agent_node` |
| Model itself calls Tavily for search | `search_agent_node` (`.bind_tools`) + `tool_node` (`ToolNode`) |
| Model has access to previous messages (tool calls, summary, quiz question) | `state["messages"]` (`add_messages` reducer), appended to by search, summary, question, and grade nodes |
| Summarization — 3–4 paragraphs, summary-only source | `summary_node` prompt |
| Quiz question answerable from summary alone | `generate_question_node` prompt |
| Grading — letter grade (A–F) with citation from summary | `grade_node` + `verify_citation()` fallback |
| State object referenced/updated by nodes | `State` TypedDict; every node returns a partial update |
| Nodes with single responsibility | separate ask / present / generate / grade nodes throughout |
| Conditional edges for restart/exit, with state reset | `route_ready`, `route_continue`, `reset_node` |
| Full workflow executes end-to-end | final `graph.invoke()` cell |

## Sanity Checks

The interactive workflow itself depends on live LLM output and patient input, so it isn't
something you can assert against meaningfully. The deterministic helper functions
underneath it are testable, though, so `main.ipynb` includes an optional
`run_sanity_checks()` cell (Section 8) that exercises them directly — no LLM calls, no
`input()`, runs in under a second.

**Coverage:** `verify_citation()` (accepts a quote that's genuinely in the summary,
rejects one that isn't, rejects text with no quote at all), `closest_sentence()` (the
fallback-citation matcher returns text actually drawn from the summary), and the two
conditional-edge routers `route_ready()` / `route_continue()` (route correctly on
`"yes"`/`"y"`/`"no"`, and `route_ready` fails safe — loops back rather than crashing — on
unrecognized input).

**Result:** 11/11 checks pass.

```
✅ PASS — verify_citation: True for a quote actually in the summary
✅ PASS — verify_citation: False for a quote not in the summary
✅ PASS — verify_citation: False when no quote is present at all
✅ PASS — closest_sentence: returns text drawn from the summary
✅ PASS — route_ready: 'yes' routes to generate_question_node
✅ PASS — route_ready: 'y' routes to generate_question_node
✅ PASS — route_ready: 'no' loops back to ready_check_node
✅ PASS — route_ready: unrecognized input fails safe (loops back, doesn't crash)
✅ PASS — route_continue: 'yes' routes to reset_node
✅ PASS — route_continue: 'y' routes to reset_node
✅ PASS — route_continue: 'no' routes to END
-------------------------------------------------------------------
11/11 checks passed
```

Not included, on purpose: the LLM-driven nodes (extraction, summarization, question
generation, grading) and the Tavily tool call. Their outputs vary by design — different
topics, different search results, different phrasing each run — so there's no fixed
expected value to assert against without building a golden dataset and an LLM-as-judge
harness, which is disproportionate to what this rubric grades. Manual end-to-end runs
(Section 9) are the appropriate check for that part of the flow.

## Known Limitations / Future Work

- One quiz question per topic (could extend to a multi-question set).
- No persistent patient profile or history — deliberate, see privacy note above.
- CLI-style interface only; a hosted/web version would need a different I/O layer
  (and, if deployed as a multi-process service, would be exactly the case where
  `interrupt()` + a real checkpointer becomes the right tool).