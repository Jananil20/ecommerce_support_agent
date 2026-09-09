"""
main.py
-------
Interactive CLI for the e-commerce support agent.

Usage:
    python main.py                     # offline mode, demo user U100
    python main.py --user U200         # offline mode, demo user U200
    AGENT_LLM_BACKEND=anthropic ANTHROPIC_API_KEY=sk-... python main.py --user U100
    AGENT_LLM_BACKEND=openai OPENAI_API_KEY=sk-...       python main.py --user U100

Try asking things like:
    "Where is my order ORD-10001?"
    "I want to return ORD-10002, the earbuds stopped working"
    "Show me some running shoes under $90"
    "Recommend something for me"
    "What's the status of RMA-XXXXXXXX?"
"""

import argparse
import sys

from core.agent import SupportAgent


def main():
    parser = argparse.ArgumentParser(description="AI E-Commerce Customer Support Agent")
    parser.add_argument("--user", default="U100", help="user_id to chat as (see data/users.json)")
    parser.add_argument("--backend", default=None,
                         help="llm backend: offline | anthropic | openai (default: env AGENT_LLM_BACKEND or offline)")
    args = parser.parse_args()

    agent = SupportAgent(user_id=args.user, backend=args.backend)

    print("=" * 60)
    print(" AI E-Commerce Support Agent  (backend: %s)" % (args.backend or "offline"))
    print(" Chatting as user_id=%s  — type 'exit' to quit" % args.user)
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            sys.exit(0)

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        reply = agent.handle_message(user_input)
        print(f"\nAgent: {reply}")


if __name__ == "__main__":
    main()
