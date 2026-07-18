"""
main.py — Jarvis desktop assistant entry point.

Responsibilities (ONLY):
  - Load the memory manager
  - Accept user input in a loop
  - Call ai.generate_response()
  - Print the reply

No AI logic. No memory logic. No LLM calls.
"""

from memory import MemoryManager
from ai import generate_response


def main() -> None:
    print("Jarvis is online. Type 'quit' or 'exit' to stop.\n")

    memory = MemoryManager()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Jarvis] Goodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "bye"}:
            print("Jarvis: Goodbye!")
            break

        response = generate_response(user_input, memory)
        print(f"\nJarvis: {response}\n")


if __name__ == "__main__":
    main()
