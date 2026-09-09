"""
demo.py
-------
Non-interactive scripted walkthrough. Run this to see tool calling and
memory working end to end without typing anything:

    python demo.py

It simulates a user (U100) across two separate "sessions" (two SupportAgent
instances) to prove long-term memory persists across sessions, while
short-term memory only lives within one session.
"""

import os
import shutil
import tempfile

# Run against a scratch copy of the seed data so repeated demo runs (which
# create a real return / mutate order status) never corrupt the shipped
# ./data JSON files. See tools/tools.py:_data_dir() and README "Tests".
_scratch_dir = tempfile.mkdtemp(prefix="agent_demo_")
_seed_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
shutil.copytree(_seed_dir, os.path.join(_scratch_dir, "data"))
os.environ["AGENT_DATA_DIR"] = os.path.join(_scratch_dir, "data")

from core.agent import SupportAgent  # noqa: E402  (import after env var set)

SEPARATOR = "-" * 60


def run_turn(agent: SupportAgent, message: str):
    print(f"\nYou: {message}")
    reply = agent.handle_message(message)
    print(f"Agent: {reply}")


def main():
    print(SEPARATOR)
    print("SESSION 1 — user U100")
    print(SEPARATOR)
    ltm_path = os.path.join(_scratch_dir, "data", "long_term_memory.json")
    session1 = SupportAgent(user_id="U100", backend="offline", long_term_path=ltm_path)

    run_turn(session1, "Hi")
    run_turn(session1, "Where is my order ORD-10001?")
    run_turn(session1, "Show me some running shoes under $90")
    run_turn(session1, "I want to return ORD-10002, one earbud stopped charging")
    run_turn(session1, "Recommend something for me")

    print("\n" + SEPARATOR)
    print("SESSION 2 (new process/session) — same user U100")
    print("Long-term memory should recall the earlier return + order from session 1.")
    print(SEPARATOR)
    session2 = SupportAgent(user_id="U100", backend="offline", long_term_path=ltm_path)
    run_turn(session2, "Hi again, any update on my stuff?")

    print("\n" + SEPARATOR)
    print("Inspect long-term memory file directly:")
    print(SEPARATOR)
    profile = session2.long_term.get_profile("U100")
    import json
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
