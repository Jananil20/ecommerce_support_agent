# AI E-Commerce Customer Support Agent

An LLM tool-calling agent for e-commerce customer support. It handles
**product queries**, **order status**, **returns**, and **recommendations**,
and uses **two-layer memory** (short-term conversation + long-term per-user
profile) to give personalized, context-aware answers across turns *and*
across sessions.

Runs **out of the box with zero API keys** using a built-in offline
rule-based planner, and can be pointed at real Claude or GPT function-calling
by setting one environment variable.

## Features

- **Tool calling** -- 8 tools the agent can invoke: `search_products`,
  `get_product_details`, `get_order_status`, `initiate_return`,
  `get_return_status`, `get_recommendations`, `get_user_profile`,
  `escalate_to_human`.
- **Memory**
  - *Short-term*: rolling window of the current conversation (in-RAM).
  - *Long-term*: JSON-persisted per-user facts, open cases (returns,
    escalations), and an interaction log -- survives across sessions/restarts.
- **Pluggable LLM backend** -- `offline` (no key needed, keyword-based
  planner), `anthropic` (Claude function calling), or `openai` (GPT
  function calling). Same agent code, same tools, same memory either way.
- **Mock backend data** -- JSON "database" of products, orders, and users
  so the whole thing runs standalone (see `data/`).

## Project structure

```
ecommerce_support_agent/
├── main.py                  # interactive CLI chat loop
├── demo.py                  # scripted demo (no typing needed) -- shows
│                             # tool calls + memory persisting across sessions
├── core/
│   ├── agent.py              # SupportAgent: the tool-calling loop
│   └── llm_client.py         # AgentLLM: offline / anthropic / openai backends
├── tools/
│   └── tools.py               # tool implementations + JSON-schema definitions
├── memory/
│   └── memory_store.py        # ShortTermMemory + LongTermMemory
├── data/
│   ├── products.json          # mock product catalog
│   ├── orders.json            # mock orders
│   ├── users.json              # mock user profiles
│   └── long_term_memory.json   # created automatically at runtime
├── tests/
│   └── test_agent.py           # pytest suite (tools, memory, end-to-end)
└── requirements.txt
```

## Quick start

```bash
cd ecommerce_support_agent

# Zero setup needed for offline mode -- pure standard library.
python main.py --user U100

# Or run the scripted, non-interactive demo:
python demo.py
```

Try asking things like:

```
Where is my order ORD-10001?
I want to return ORD-10002, one earbud stopped charging
Show me some running shoes under $90
Recommend something for me
What's the status of RMA-XXXXXXXX?
```

Demo users available out of the box: `U100` (Priya, Gold tier) and `U200`
(Arjun, Silver tier) -- see `data/users.json`. Sample order IDs to try:
`ORD-10001`, `ORD-10002`, `ORD-10003`, `ORD-10004` (see `data/orders.json`).

## Using a real LLM backend (optional)

The offline planner is a small keyword matcher meant to demonstrate the
full pipeline without credentials -- swap in a real model for genuinely
robust natural-language understanding:

```bash
pip install anthropic          # or: pip install openai

# Claude:
export AGENT_LLM_BACKEND=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
python main.py --user U100

# GPT:
export AGENT_LLM_BACKEND=openai
export OPENAI_API_KEY=sk-...
python main.py --user U100
```

`core/llm_client.py` normalizes both providers' function-calling responses
into the same `{"type": "tool_calls" | "text", ...}` shape that
`core/agent.py` consumes, so the rest of the system is backend-agnostic.

## How memory works

- **Short-term** (`ShortTermMemory`): a rolling in-memory buffer of the last
  N turns for the *current* session, passed back to the LLM each turn so it
  has conversational context. Cleared when the process ends.
- **Long-term** (`LongTermMemory`): persisted to
  `data/long_term_memory.json`, keyed by `user_id`. Stores:
  - `facts` -- e.g. `last_order_discussed`, `last_search_query`
  - `open_cases` -- e.g. an in-progress return (RMA) or support escalation
  - `interaction_log` -- a bounded rolling log of past turns' summaries

  Every `SupportAgent` call injects a short context blurb built from long-term
  memory into the prompt, so returning users get continuity even in a brand
  new process -- see `demo.py` for a two-session walkthrough that proves this.

## Extending it

- **Add a tool**: implement a function in `tools/tools.py`, register it in
  `TOOL_REGISTRY`, and add its JSON-schema entry to `TOOL_SCHEMAS`. Both
  the offline planner and real LLM backends pick it up automatically.
- **Swap mock data for a real system**: replace the JSON reads/writes in
  `tools/tools.py` with calls to your actual order-management/product API --
  the tool function signatures and the rest of the agent don't need to change.
- **Swap memory storage**: `LongTermMemory` exposes a small public API
  (`get_profile`, `set_fact`, `add_open_case`, `log_interaction`, ...) --
  reimplement its internals against Redis/Postgres/a vector DB without
  touching `core/agent.py`.

## Tests

```bash
pip install pytest
pytest -q
```
