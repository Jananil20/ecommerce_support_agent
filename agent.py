from __future__ import annotations
import json
from typing import Any, Dict, List, Optional

from core.llm_client import AgentLLM
from memory.memory_store import ShortTermMemory, LongTermMemory
from tools.tools import call_tool

MAX_TOOL_HOPS = 4


class SupportAgent:
    def __init__(self, user_id: str, backend: Optional[str] = None,
                 long_term_path: str = "data/long_term_memory.json"):
        self.user_id = user_id
        self.llm = AgentLLM(backend=backend)
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory(path=long_term_path)

    # ------------------------------------------------------------------
    def _context_preamble(self) -> str:
        """Summarize long-term memory into a short context blurb for the LLM."""
        profile = self.long_term.get_profile(self.user_id)
        facts = profile.get("facts", {})
        open_cases = profile.get("open_cases", [])
        recent = self.long_term.recent_interactions(self.user_id, n=3)

        parts = [f"[Context for user_id={self.user_id}]"]
        if facts:
            parts.append("Known facts: " + json.dumps(facts))
        if open_cases:
            parts.append("Open cases: " + json.dumps(open_cases))
        if recent:
            summaries = "; ".join(r["summary"] for r in recent)
            parts.append("Recent past interactions: " + summaries)
        return "\n".join(parts)

    # ------------------------------------------------------------------
    def handle_message(self, user_message: str) -> str:
        self.short_term.add("user", user_message)

        # Build the message list the LLM backend expects: plain role/content
        # turns, with a context system-ish note prepended as the first user turn
        # (kept simple/backend-agnostic rather than using a separate system slot
        # per-call, since offline mode has no notion of 'system').
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": self._context_preamble()},
            {"role": "assistant", "content": "Understood, I'll use that context."},
        ]
        messages.extend(
            {"role": t["role"], "content": t["content"]}
            for t in self.short_term.as_list()
            if t["role"] in ("user", "assistant")
        )

        final_text = self._run_tool_loop(messages)

        self.short_term.add("assistant", final_text)
        self.long_term.log_interaction(self.user_id, f"user: {user_message[:120]} | agent: {final_text[:120]}")
        return final_text

    # ------------------------------------------------------------------
    def _run_tool_loop(self, messages: List[Dict[str, Any]]) -> str:
        for _ in range(MAX_TOOL_HOPS):
            step = self.llm.next_step(messages)

            if step["type"] == "text":
                return step["content"]

            # step["type"] == "tool_calls"
            for call in step["calls"]:
                result = call_tool(call["name"], call["arguments"])
                self._remember_side_effects(call["name"], call["arguments"], result)

                # Feed the tool call + result back into the running transcript.
                messages.append({"role": "assistant",
                                  "content": f"[calling tool {call['name']} with {json.dumps(call['arguments'])}]"})
                messages.append({"role": "tool", "name": call["name"], "content": json.dumps(result)})

        return "I'm having trouble completing that request right now — let me connect you with a human agent."

    # ------------------------------------------------------------------
    def _remember_side_effects(self, tool_name: str, args: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Persist durable facts learned from a tool call into long-term memory."""
        if tool_name == "initiate_return" and result.get("success"):
            rma = result["rma"]
            self.long_term.add_open_case(self.user_id, {
                "case_id": rma["rma_id"],
                "type": "return",
                "order_id": rma["order_id"],
                "status": rma["status"],
            })
        if tool_name == "get_order_status" and result.get("order"):
            self.long_term.set_fact(self.user_id, "last_order_discussed", result["order"]["order_id"])
        if tool_name == "search_products":
            q = args.get("query")
            if q:
                self.long_term.set_fact(self.user_id, "last_search_query", q)
        if tool_name == "escalate_to_human" and result.get("success"):
            self.long_term.add_open_case(self.user_id, {
                "case_id": result["ticket_id"],
                "type": "escalation",
                "status": "open",
                "summary": args.get("issue_summary", ""),
            })
