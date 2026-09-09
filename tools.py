"""
tools.py
--------
The actual callable "tools" the agent exposes to the LLM (or to the
rule-based fallback planner). Each tool:
  - has a JSON-schema description (TOOL_SCHEMAS) for function-calling APIs
  - has a plain python implementation registered in TOOL_REGISTRY

Backed by simple JSON files in /data to simulate a product catalog,
an order-management system, and a returns/RMA system.
"""

from __future__ import annotations
import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data")


def _data_dir() -> str:
    """
    Resolved at call time (not import time) so callers -- e.g. tests, or a
    demo run that wants a scratch copy -- can point AGENT_DATA_DIR at an
    isolated directory instead of mutating the shipped seed data in ./data.
    """
    return os.environ.get("AGENT_DATA_DIR", DEFAULT_DATA_DIR)


def _load(name: str) -> Any:
    with open(os.path.join(_data_dir(), name), "r", encoding="utf-8") as f:
        return json.load(f)


def _save_orders(orders: List[Dict[str, Any]]) -> None:
    with open(os.path.join(_data_dir(), "orders.json"), "w", encoding="utf-8") as f:
        json.dump(orders, f, indent=2)


def _save_returns(returns: List[Dict[str, Any]]) -> None:
    with open(os.path.join(_data_dir(), "returns.json"), "w", encoding="utf-8") as f:
        json.dump(returns, f, indent=2)


def _load_returns() -> List[Dict[str, Any]]:
    path = os.path.join(_data_dir(), "returns.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "some", "any", "me", "please", "show", "find", "search",
    "for", "looking", "look", "i", "want", "need", "do", "you", "have", "got",
    "under", "below", "over", "above", "with", "and", "or", "of", "to", "in",
    "buy", "get", "can", "could", "would", "like", "similar", "recommend",
}


def _tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


def search_products(query: str = "", category: str = "", max_price: float = None,
                     min_rating: float = None) -> Dict[str, Any]:
    """Search the product catalog by free-text query, category, price ceiling, min rating."""
    products = _load("products.json")
    tokens = _tokenize(query or "")
    results = []
    for p in products:
        if tokens:
            haystack = " ".join([p["name"], p["category"], p["description"], " ".join(p["tags"])]).lower()
            # match if ANY meaningful token from the query appears in the product's text
            if not any(tok in haystack for tok in tokens):
                continue
        if category and p["category"].lower() != category.lower():
            continue
        if max_price is not None and p["price"] > max_price:
            continue
        if min_rating is not None and p["rating"] < min_rating:
            continue
        results.append(p)
    return {"count": len(results), "products": results}


def get_product_details(product_id: str) -> Dict[str, Any]:
    """Fetch full details for a single product by ID."""
    products = _load("products.json")
    for p in products:
        if p["id"].lower() == product_id.lower():
            return {"found": True, "product": p}
    return {"found": False, "error": f"No product found with id {product_id}"}


def get_order_status(order_id: str = "", user_id: str = "") -> Dict[str, Any]:
    """Look up order status by order_id, or list all orders for a user_id."""
    orders = _load("orders.json")
    if order_id:
        for o in orders:
            if o["order_id"].lower() == order_id.lower():
                return {"found": True, "order": o}
        return {"found": False, "error": f"No order found with id {order_id}"}
    if user_id:
        matches = [o for o in orders if o["user_id"] == user_id]
        return {"found": bool(matches), "orders": matches}
    return {"found": False, "error": "Provide order_id or user_id"}


def initiate_return(order_id: str, reason: str, user_id: str = "") -> Dict[str, Any]:
    """Start a return/RMA for an order if it is within the return window and eligible."""
    orders = _load("orders.json")
    order = next((o for o in orders if o["order_id"].lower() == order_id.lower()), None)
    if not order:
        return {"success": False, "error": f"No order found with id {order_id}"}
    if user_id and order["user_id"] != user_id:
        return {"success": False, "error": "This order does not belong to the given user."}
    if not order.get("return_eligible", False):
        return {
            "success": False,
            "error": f"Order {order_id} is not eligible for return "
                     f"(status: {order['status']}). Items must be delivered and within the "
                     f"{order.get('return_window_days', 30)}-day return window."
        }

    returns = _load_returns()
    rma_id = f"RMA-{uuid.uuid4().hex[:8].upper()}"
    record = {
        "rma_id": rma_id,
        "order_id": order["order_id"],
        "user_id": order["user_id"],
        "reason": reason,
        "status": "Approved - awaiting drop-off",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "refund_amount": order["total"],
    }
    returns.append(record)
    _save_returns(returns)

    # mark the order as return-initiated so it can't be double-requested
    for o in orders:
        if o["order_id"] == order["order_id"]:
            o["return_eligible"] = False
            o["status"] = "Return Initiated"
    _save_orders(orders)

    return {"success": True, "rma": record}


def get_return_status(rma_id: str) -> Dict[str, Any]:
    """Check the status of a previously initiated return by RMA id."""
    returns = _load_returns()
    for r in returns:
        if r["rma_id"].lower() == rma_id.lower():
            return {"found": True, "return": r}
    return {"found": False, "error": f"No return found with id {rma_id}"}


def get_recommendations(user_id: str = "", category: str = "", based_on_product_id: str = "",
                         top_n: int = 3) -> Dict[str, Any]:
    """
    Recommend products. Strategy (in priority order):
      1. If based_on_product_id given -> similar items (same category, excluding itself), best rated first.
      2. Else if user_id given -> use stored preference categories from the user profile.
      3. Else if category given -> top rated items in that category.
      4. Else -> overall top-rated items.
    """
    products = _load("products.json")
    users = _load("users.json")

    def top_rated(pool, n):
        return sorted(pool, key=lambda p: (-p["rating"], p["price"]))[:n]

    if based_on_product_id:
        base = next((p for p in products if p["id"].lower() == based_on_product_id.lower()), None)
        if not base:
            return {"error": f"No product found with id {based_on_product_id}"}
        pool = [p for p in products if p["category"] == base["category"] and p["id"] != base["id"]]
        return {"strategy": "similar_to_product", "base_product": base["name"],
                "recommendations": top_rated(pool, top_n)}

    if user_id:
        user = next((u for u in users if u["user_id"] == user_id), None)
        if user:
            cats = user.get("preferences", {}).get("categories", [])
            pool = [p for p in products if p["category"] in cats]
            if pool:
                return {"strategy": "user_preference", "categories_used": cats,
                        "recommendations": top_rated(pool, top_n)}

    if category:
        pool = [p for p in products if p["category"].lower() == category.lower()]
        return {"strategy": "category_top_rated", "recommendations": top_rated(pool, top_n)}

    return {"strategy": "overall_top_rated", "recommendations": top_rated(products, top_n)}


def get_user_profile(user_id: str) -> Dict[str, Any]:
    """Fetch a user's stored profile (name, loyalty tier, preferences)."""
    users = _load("users.json")
    for u in users:
        if u["user_id"] == user_id:
            return {"found": True, "user": u}
    return {"found": False, "error": f"No user found with id {user_id}"}


def escalate_to_human(user_id: str, issue_summary: str, priority: str = "normal") -> Dict[str, Any]:
    """Escalate the conversation to a human support agent, e.g. for complex or sensitive issues."""
    ticket_id = f"TCK-{uuid.uuid4().hex[:8].upper()}"
    return {
        "success": True,
        "ticket_id": ticket_id,
        "priority": priority,
        "message": f"A human support agent will follow up on ticket {ticket_id} regarding: {issue_summary}",
    }


# ---------------------------------------------------------------------------
# Registry + JSON-schema definitions for function-calling APIs
# (Compatible with both OpenAI's `tools` param and Anthropic's `tools` param
#  with minor key renaming done in core/llm_client.py)
# ---------------------------------------------------------------------------

TOOL_REGISTRY = {
    "search_products": search_products,
    "get_product_details": get_product_details,
    "get_order_status": get_order_status,
    "initiate_return": initiate_return,
    "get_return_status": get_return_status,
    "get_recommendations": get_recommendations,
    "get_user_profile": get_user_profile,
    "escalate_to_human": escalate_to_human,
}

TOOL_SCHEMAS = [
    {
        "name": "search_products",
        "description": "Search the product catalog by free text query, category, max price, and/or min rating.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Free text search, e.g. 'wireless earbuds'"},
                "category": {"type": "string", "description": "e.g. Footwear, Electronics, Kitchen, Bags, Home"},
                "max_price": {"type": "number", "description": "Maximum price filter"},
                "min_rating": {"type": "number", "description": "Minimum rating filter (0-5)"},
            },
            "required": [],
        },
    },
    {
        "name": "get_product_details",
        "description": "Get full details for one product given its product_id.",
        "parameters": {
            "type": "object",
            "properties": {"product_id": {"type": "string"}},
            "required": ["product_id"],
        },
    },
    {
        "name": "get_order_status",
        "description": "Get the status/tracking of an order by order_id, or list all orders for a user_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "user_id": {"type": "string"},
            },
            "required": [],
        },
    },
    {
        "name": "initiate_return",
        "description": "Start a return/RMA for a delivered, return-eligible order.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "reason": {"type": "string", "description": "Why the customer wants to return the item"},
                "user_id": {"type": "string"},
            },
            "required": ["order_id", "reason"],
        },
    },
    {
        "name": "get_return_status",
        "description": "Check the status of a previously created return by rma_id.",
        "parameters": {
            "type": "object",
            "properties": {"rma_id": {"type": "string"}},
            "required": ["rma_id"],
        },
    },
    {
        "name": "get_recommendations",
        "description": "Recommend products for a user, similar to a given product, or top items in a category.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "category": {"type": "string"},
                "based_on_product_id": {"type": "string"},
                "top_n": {"type": "integer", "description": "Number of recommendations, default 3"},
            },
            "required": [],
        },
    },
    {
        "name": "get_user_profile",
        "description": "Fetch a user's stored profile: name, loyalty tier, category preferences.",
        "parameters": {
            "type": "object",
            "properties": {"user_id": {"type": "string"}},
            "required": ["user_id"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Escalate a complex, sensitive, or unresolved issue to a human support agent.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "issue_summary": {"type": "string"},
                "priority": {"type": "string", "enum": ["low", "normal", "high", "urgent"]},
            },
            "required": ["user_id", "issue_summary"],
        },
    },
]


def call_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch a tool call by name with a dict of arguments; returns JSON-safe dict."""
    if name not in TOOL_REGISTRY:
        return {"error": f"Unknown tool '{name}'"}
    fn = TOOL_REGISTRY[name]
    try:
        return fn(**arguments)
    except TypeError as e:
        return {"error": f"Bad arguments for tool '{name}': {e}"}
    except Exception as e:  # pragma: no cover - defensive
        return {"error": f"Tool '{name}' raised an exception: {e}"}
