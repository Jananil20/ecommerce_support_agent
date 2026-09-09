"""
Basic tests covering tools, memory, and the end-to-end offline agent loop.
Run with:  pytest -q

Tests that mutate order/return state point AGENT_DATA_DIR at a fresh copy
of the seed data (via the `isolated_data_dir` fixture) so they never touch
-- or corrupt -- the real ./data seed files shipped with the project.
"""
import json
import os
import shutil
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED_DATA_DIR = os.path.join(REPO_ROOT, "data")


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    """Copy the seed JSON data to a temp dir and point tools at it for this test only."""
    scratch = tmp_path / "data"
    shutil.copytree(SEED_DATA_DIR, scratch, ignore=shutil.ignore_patterns("long_term_memory.json", "returns.json"))
    monkeypatch.setenv("AGENT_DATA_DIR", str(scratch))
    return str(scratch)


from tools.tools import search_products, get_order_status, initiate_return, get_recommendations
from memory.memory_store import ShortTermMemory, LongTermMemory
from core.agent import SupportAgent


def test_search_products_by_category():
    res = search_products(category="Footwear")
    assert res["count"] >= 2
    assert all(p["category"] == "Footwear" for p in res["products"])


def test_search_products_max_price():
    res = search_products(category="Electronics", max_price=60)
    assert all(p["price"] <= 60 for p in res["products"])


def test_get_order_status_found():
    res = get_order_status(order_id="ORD-10001")
    assert res["found"] is True
    assert res["order"]["order_id"] == "ORD-10001"


def test_get_order_status_not_found():
    res = get_order_status(order_id="ORD-99999")
    assert res["found"] is False


def test_initiate_return_ineligible_order(isolated_data_dir):
    res = initiate_return(order_id="ORD-10003", reason="changed my mind")
    assert res["success"] is False


def test_initiate_return_eligible_order(isolated_data_dir):
    res = initiate_return(order_id="ORD-10002", reason="arrived damaged", user_id="U100")
    assert res["success"] is True
    assert res["rma"]["order_id"] == "ORD-10002"
    # a second attempt on the same order should now fail (already returned)
    res2 = initiate_return(order_id="ORD-10002", reason="again", user_id="U100")
    assert res2["success"] is False


def test_get_recommendations_by_category():
    res = get_recommendations(category="Kitchen", top_n=2)
    assert len(res["recommendations"]) <= 2
    assert all(p["category"] == "Kitchen" for p in res["recommendations"])


def test_short_term_memory_rolling_window():
    mem = ShortTermMemory(max_turns=3)
    for i in range(5):
        mem.add("user", f"msg {i}")
    assert len(mem.as_list()) == 3
    assert mem.as_list()[0]["content"] == "msg 2"


def test_long_term_memory_persists_across_instances():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ltm.json")
        ltm1 = LongTermMemory(path=path)
        ltm1.set_fact("U999", "favorite_category", "Electronics")
        ltm1.add_open_case("U999", {"case_id": "RMA-TEST1", "type": "return", "status": "pending"})

        ltm2 = LongTermMemory(path=path)  # simulate a new session/process
        profile = ltm2.get_profile("U999")
        assert profile["facts"]["favorite_category"] == "Electronics"
        assert profile["open_cases"][0]["case_id"] == "RMA-TEST1"


def test_agent_end_to_end_order_status(tmp_path):
    ltm_path = str(tmp_path / "ltm.json")
    agent = SupportAgent(user_id="U100", backend="offline", long_term_path=ltm_path)
    reply = agent.handle_message("Where is my order ORD-10001?")
    assert "ORD-10001" in reply
    assert "Shipped" in reply or "shipped" in reply.lower()

    # long-term memory should now remember the last order discussed
    fact = agent.long_term.get_fact("U100", "last_order_discussed")
    assert fact == "ORD-10001"


def test_agent_end_to_end_return_flow(tmp_path, isolated_data_dir):
    ltm_path = str(tmp_path / "ltm.json")
    agent = SupportAgent(user_id="U100", backend="offline", long_term_path=ltm_path)
    reply = agent.handle_message("I want to return ORD-10002, it arrived damaged")
    assert "RMA-" in reply

    profile = agent.long_term.get_profile("U100")
    assert len(profile["open_cases"]) == 1
    assert profile["open_cases"][0]["type"] == "return"
