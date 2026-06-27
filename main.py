"""
main.py
-------
CLI entry point. Run: python main.py
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()  # reads .env into os.environ before agent.py / tools.py read keys

from agent import run_agent


def main():
    print("=" * 60)
    print("YouTube AI Research Agent")
    print("=" * 60)
    print("Enter an AI-related topic (e.g. 'Claude vs GPT-4').")
    print("Type 'quit' to exit.\n")

    # Fail fast with a clear message if keys are missing, rather than
    # letting the agent loop fail confusingly several steps in.
    missing = [k for k in ("YOUTUBE_API_KEY", "GEMINI_API_KEY") if not os.environ.get(k)]
    if missing:
        print(f"Missing environment variable(s): {', '.join(missing)}")
        print("Add them to a .env file (see .env.example) before running.\n")
        sys.exit(1)

    while True:
        topic = input("Topic> ").strip()
        if topic.lower() in ("quit", "exit"):
            break
        if not topic:
            continue

        print(f"\nResearching '{topic}'...\n")
        brief = run_agent(topic, verbose=True)

        print("\n" + "=" * 60)
        print("FINAL BRIEF")
        print("=" * 60)
        print(brief)
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
