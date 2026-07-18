"""
LangChain + Groq Conversational Chatbot
Maintains full conversation history, persisted to memory.json
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# ── Config ────────────────────────────────────────────────────────────────────
MEMORY_FILE = Path(__file__).parent / "memory.json"
MODEL_NAME  = "llama-3.1-8b-instant"

# ── Load environment ──────────────────────────────────────────────────────────
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise EnvironmentError("GROQ_API_KEY not found. Add it to your .env file.")

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model=MODEL_NAME,
    api_key=api_key,
    # Easy to extend: add temperature=0.7, max_tokens=1024, etc.
)

# ── Memory helpers ────────────────────────────────────────────────────────────
def load_history() -> list[dict]:
    """Load previous messages from disk, or return empty list."""
    if MEMORY_FILE.exists():
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_history(history: list[dict]) -> None:
    """Persist conversation history to disk."""
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def to_langchain_messages(history: list[dict]) -> list[BaseMessage]:
    """Convert stored dicts → LangChain message objects."""
    messages = []
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages


# ── Chat loop ─────────────────────────────────────────────────────────────────
def main() -> None:
    history = load_history()

    if history:
        print(f"[Loaded {len(history)} messages from memory.json]\n")

    print("Groq Chatbot  |  type 'exit' or 'quit' to stop\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # Append user message to history
        history.append({"role": "user", "content": user_input})

        # Build full message list and call the model
        lc_messages = to_langchain_messages(history)
        response = llm.invoke(lc_messages)
        reply = response.content

        # Append assistant reply and persist
        history.append({"role": "assistant", "content": reply})
        save_history(history)

        print(f"\nAssistant: {reply}\n")


if __name__ == "__main__":
    main()
