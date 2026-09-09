"""
llm_client.py
-------------
Thin abstraction so the agent loop doesn't care which LLM backend issues
the tool calls. Three backends are supported:

  - "anthropic": real Claude function-calling via the Anthropic SDK
  - "openai":    real function-calling via the OpenAI SDK
  - "offline":   a small built-in rule-based intent matcher that needs NO
                 API key at all, so the whole project runs out of the box.
                 It picks a tool (or none) using keyword rules, which is
                 obviously far less capable than a real LLM but is enough
                 to demonstrate the full tool-calling + memory pipeline
                 end to end without any credentials.

Set the backend with the AGENT_LLM_BACKEND env var, or pass it explicitly
to AgentLLM(). Default: "offline".
"""

from __future__ import annotations
import os
import re
import json
from typing import Any, Dict, List, Optional

from tools.tools import TOOL_SCHEMAS


SYSTEM_PROMPT = """You are a helpful, concise AI customer support agent for an e-commerce store.
You can call tools to search products, check order status, start returns, and give
recommendations. Always call a tool when the user's request needs live data (orders,
stock, prices) rather than guessing. Use the conversation memory you're given about the
user's past interactions to personalize answers. Keep answers short, friendly, and
actionable. If something is outside what your tools can resolve (e.g. a fraud claim or
a billing dispute), use the escalate_to_human tool."""


class AgentLLM:
    def __init__(self, backend: Optional[str] = None, model: Optional[str] = None):
        self.backend = backend or os.environ.get("AGENT_LLM_BACKEND", "offline")
        self.model = model

        if self.backend == "anthropic":
            import anthropic  # requires ANTHROPIC_API_KEY in env
            self.client = anthropic.Anthropic()
            self.model = self.model or "claude-sonnet-4-6"
        elif self.backend == "openai":
            import openai  # requires OPENAI_API_KEY in env
            self.client = openai.OpenAI()
            self.model = self.model or "gpt-4o"
        elif self.backend == "offline":
            self.client = None
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    # ------------------------------------------------------------------
    # Public entry point used by the agent loop.
    # Returns a normalized dict:
    #   {"type": "tool_calls", "calls": [{"id":..., "name":..., "arguments": {...}}]}
    #   {"type": "text", "content": "final reply"}
    # ------------------------------------------------------------------
    def next_step(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.backend == "anthropic":
            return self._step_anthropic(messages)
        if self.backend == "openai":
            return self._step_openai(messages)
        return self._step_offline(messages)

    # ------------------------------------------------------------------
    # Anthropic backend
    # ------------------------------------------------------------------
    def _step_anthropic(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        anthropic_tools = [
            {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
            for t in TOOL_SCHEMAS
        ]
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=anthropic_tools,
            messages=messages,
        )
        calls = [
            {"id": b.id, "name": b.name, "arguments": b.input}
            for b in resp.content if b.type == "tool_use"
        ]
        if calls:
            return {"type": "tool_calls", "calls": calls, "raw_content": resp.content}
        text = "".join(b.text for b in resp.content if b.type == "text")
        return {"type": "text", "content": text}

    # ------------------------------------------------------------------
    # OpenAI backend
    # ------------------------------------------------------------------
    def _step_openai(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        openai_tools = [
            {"type": "function", "function": {"name": t["name"], "description": t["description"],
                                               "parameters": t["parameters"]}}
            for t in TOOL_SCHEMAS
        ]
        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        resp = self.client.chat.completions.create(
            model=self.model, messages=full_messages, tools=openai_tools,
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            calls = [
                {"id": tc.id, "name": tc.function.name, "arguments": json.loads(tc.function.arguments)}
                for tc in msg.tool_calls
            ]
            return {"type": "tool_calls", "calls": calls, "raw_content": msg}
        return {"type": "text", "content": msg.content or ""}

    # ------------------------------------------------------------------
    # Offline rule-based backend (no API key needed)
    # ------------------------------------------------------------------
    def _step_offline(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Very small keyword/regex planner so the demo works without any API key.
        It looks only at the latest user message (plus lightweight context)
        and picks at most one tool call. If a tool result was just returned
        (last message role == 'tool'), it synthesizes a final text answer
        instead of calling another tool.
        """
        last = messages[-1]

        # If we just got a tool result back, summarize it in plain text.
        if last["role"] == "tool":
            return {"type": "text", "content": self._offline_summarize(messages)}

        user_text = ""
        for m in reversed(messages):
            if m["role"] == "user":
                user_text = m["content"]
                break
        text = user_text.lower()

        def find(pattern):
            m = re.search(pattern, user_text, re.IGNORECASE)
            return m.group(1).strip() if m else None

        order_id = find(r"\b(ORD-\d+)\b")
        rma_id = find(r"\b(RMA-[A-Z0-9]+)\b")
        user_id = find(r"\b(U\d+)\b")

        if rma_id and ("status" in text or "return" in text or "rma" in text):
            return {"type": "tool_calls",
                    "calls": [{"id": "offline-1", "name": "get_return_status", "arguments": {"rma_id": rma_id}}]}

        if "return" in text or "refund" in text or "send back" in text:
            if order_id:
                reason = user_text
                return {"type": "tool_calls",
                        "calls": [{"id": "offline-1", "name": "initiate_return",
                                   "arguments": {"order_id": order_id, "reason": reason,
                                                 "user_id": user_id or ""}}]}
            return {"type": "text",
                    "content": "I can help start a return — could you share the order ID (e.g. ORD-10001)?"}

        if order_id or "order status" in text or "where is my order" in text or "track" in text:
            args = {}
            if order_id:
                args["order_id"] = order_id
            elif user_id:
                args["user_id"] = user_id
            if args:
                return {"type": "tool_calls",
                        "calls": [{"id": "offline-1", "name": "get_order_status", "arguments": args}]}
            return {"type": "text",
                    "content": "Sure — what's your order ID (e.g. ORD-10001) or your user ID?"}

        if any(k in text for k in ["recommend", "suggest", "what should i buy", "similar to"]):
            args: Dict[str, Any] = {}
            pid = find(r"\b(P\d+)\b")
            if pid:
                args["based_on_product_id"] = pid
            if user_id:
                args["user_id"] = user_id
            for cat in ["footwear", "electronics", "kitchen", "bags", "home"]:
                if cat in text:
                    args["category"] = cat.capitalize()
            return {"type": "tool_calls",
                    "calls": [{"id": "offline-1", "name": "get_recommendations", "arguments": args}]}

        if any(k in text for k in ["find", "search", "looking for", "show me", "do you have", "buy"]):
            args: Dict[str, Any] = {"query": user_text}
            for cat in ["footwear", "electronics", "kitchen", "bags", "home"]:
                if cat in text:
                    args["category"] = cat.capitalize()
            price_match = re.search(r"under\s*\$?(\d+(?:\.\d+)?)", text)
            if price_match:
                args["max_price"] = float(price_match.group(1))
            return {"type": "tool_calls",
                    "calls": [{"id": "offline-1", "name": "search_products", "arguments": args}]}

        if any(k in text for k in ["human", "agent", "manager", "complaint", "frustrated", "escalate"]):
            return {"type": "tool_calls",
                    "calls": [{"id": "offline-1", "name": "escalate_to_human",
                               "arguments": {"user_id": user_id or "UNKNOWN",
                                             "issue_summary": user_text, "priority": "high"}}]}

        if any(k in text for k in ["hi", "hello", "hey"]) and len(text) < 20:
            return {"type": "text", "content": "Hi! I'm your shopping assistant — I can check orders, "
                                                "start returns, find products, or recommend something. "
                                                "What do you need help with?"}

        return {"type": "text",
                "content": "I can help with product search, order status, returns, or recommendations. "
                            "Could you tell me a bit more about what you need?"}

    def _offline_summarize(self, messages: List[Dict[str, Any]]) -> str:
        """Turn the most recent tool result into a short human-readable reply."""
        last_tool_msg = messages[-1]
        try:
            data = json.loads(last_tool_msg["content"])
        except Exception:
            return "Here's what I found: " + str(last_tool_msg["content"])

        name = last_tool_msg.get("name", "")

        if name == "get_order_status":
            if data.get("order"):
                o = data["order"]
                lines = [f"Order {o['order_id']}: status **{o['status']}**."]
                if o.get("tracking_number"):
                    lines.append(f"Tracking: {o['tracking_number']} via {o.get('carrier','carrier')}.")
                if o.get("expected_delivery"):
                    lines.append(f"Expected delivery: {o['expected_delivery']}.")
                return " ".join(lines)
            if data.get("orders"):
                return "Here are your orders: " + "; ".join(
                    f"{o['order_id']} ({o['status']})" for o in data["orders"])
            return data.get("error", "I couldn't find that order.")

        if name == "initiate_return":
            if data.get("success"):
                r = data["rma"]
                return (f"Your return has been created: {r['rma_id']} for order {r['order_id']}. "
                        f"Status: {r['status']}. Estimated refund: ${r['refund_amount']}.")
            return data.get("error", "I couldn't start that return.")

        if name == "get_return_status":
            if data.get("found"):
                r = data["return"]
                return f"Return {r['rma_id']} is currently: {r['status']}."
            return data.get("error", "I couldn't find that return.")

        if name == "search_products":
            prods = data.get("products", [])
            if not prods:
                return "I couldn't find any matching products — want to try a different search?"
            lines = [f"{p['name']} — ${p['price']} (rating {p['rating']}/5, {p['stock']} in stock)"
                     for p in prods[:5]]
            return "Here's what I found:\n- " + "\n- ".join(lines)

        if name == "get_recommendations":
            recs = data.get("recommendations", [])
            if not recs:
                return "I don't have a strong recommendation yet — tell me a category you like."
            lines = [f"{p['name']} — ${p['price']} (rating {p['rating']}/5)" for p in recs]
            return "You might like:\n- " + "\n- ".join(lines)

        if name == "escalate_to_human":
            return data.get("message", "I've escalated this to a human agent.")

        if name == "get_user_profile":
            if data.get("found"):
                u = data["user"]
                return f"{u['name']} — {u['loyalty_tier']} tier."
            return data.get("error", "I couldn't find that user.")

        return "Here's what I found: " + json.dumps(data)
