"""
memory.py — Three-layer memory manager for the Jarvis AI backend.

Architecture:
  Layer 1 — recent.json     : Last N raw messages (sliding window)
  Layer 2 — summaries.json  : Compressed conversation summaries
  Layer 3 — profile.json    : Durable user facts (preferences, projects, goals)

Design goal: send the *minimum* tokens needed to give the model enough context,
while never losing important information across sessions.

v2 change: build_messages() now returns list[BaseMessage] (native LangChain objects)
instead of a single flat string. This enables proper tool-calling message sequences
and keeps memory of past tool calls intact.
"""

import json
import os
from datetime import datetime
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# ── Constants ──────────────────────────────────────────────────────────────────

MEMORY_DIR = os.path.join(os.path.dirname(__file__), "Memory")

RECENT_PATH    = os.path.join(MEMORY_DIR, "recent.json")
SUMMARIES_PATH = os.path.join(MEMORY_DIR, "summaries.json")
PROFILE_PATH   = os.path.join(MEMORY_DIR, "profile.json")

# How many messages to keep in the sliding window sent to the model
RECENT_WINDOW: int = 10

# Trigger a summary after this many total stored messages (~20 exchanges)
SUMMARY_TRIGGER: int = 40

# How many summaries to include in each prompt
MAX_SUMMARIES_IN_PROMPT: int = 3

# Compress when we have more summaries than this
SUMMARY_COMPRESS_THRESHOLD: int = 6


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ensure_files() -> None:
    """Create the Memory directory and empty JSON files if they don't exist."""
    os.makedirs(MEMORY_DIR, exist_ok=True)
    for path, default in [
        (RECENT_PATH,    []),
        (SUMMARIES_PATH, []),
        (PROFILE_PATH,   {}),
    ]:
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(default, f, indent=2)


def _read_json(path: str) -> Any:
    """Read and return JSON from a file. Returns empty list/dict on error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return [] if path != PROFILE_PATH else {}


def _write_json(path: str, data: Any) -> None:
    """Atomically write data as JSON."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)  # atomic on POSIX; near-atomic on Windows


# ── MemoryManager ──────────────────────────────────────────────────────────────

class MemoryManager:
    """
    Manages all three memory layers.

    Public API (called from ai.py and main.py):
      add_message(role, content)   — append a raw message and persist immediately
      build_context()              — return the prompt context string
      get_recent_messages()        — return the raw recent messages list
      update_profile(updates)      — merge new facts into profile.json
      summarize_and_compress(llm)  — summarise + optionally compress (called from ai.py)
      should_summarize()           — True when recent.json has SUMMARY_TRIGGER messages
    """

    def __init__(self) -> None:
        _ensure_files()

    # ── Layer 1: Recent messages ───────────────────────────────────────────────

    def add_message(self, role: str, content: str) -> None:
        """
        Append one message to recent.json immediately.
        Persistence is immediate so a crash never loses a message.
        """
        messages: list[dict] = _read_json(RECENT_PATH)
        messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        })
        _write_json(RECENT_PATH, messages)

    def get_recent_messages(self) -> list[dict]:
        """Return the full recent message list (used for summarisation decisions)."""
        return _read_json(RECENT_PATH)

    def _get_recent_window(self) -> list[dict]:
        """Return only the last RECENT_WINDOW messages for the prompt."""
        messages = _read_json(RECENT_PATH)
        return messages[-RECENT_WINDOW:]

    def should_summarize(self) -> bool:
        """Return True when recent.json has accumulated enough messages."""
        return len(_read_json(RECENT_PATH)) >= SUMMARY_TRIGGER

    # ── Layer 2: Summaries ─────────────────────────────────────────────────────

    def append_summary(self, summary_text: str) -> None:
        """Add a new summary entry and clear recent.json."""
        summaries: list[dict] = _read_json(SUMMARIES_PATH)
        summaries.append({
            "summary": summary_text,
            "created_at": datetime.utcnow().isoformat(),
        })
        _write_json(SUMMARIES_PATH, summaries)
        _write_json(RECENT_PATH, [])  # clear after summarising

    def get_recent_summaries(self) -> list[dict]:
        """Return the most recent MAX_SUMMARIES_IN_PROMPT summaries."""
        summaries: list[dict] = _read_json(SUMMARIES_PATH)
        return summaries[-MAX_SUMMARIES_IN_PROMPT:]

    def needs_compression(self) -> bool:
        """True when we have too many summaries and should merge old ones."""
        return len(_read_json(SUMMARIES_PATH)) > SUMMARY_COMPRESS_THRESHOLD

    def compress_summaries(self, compressed_text: str) -> None:
        """
        Replace old summaries with a single compressed one.
        Keeps the most recent MAX_SUMMARIES_IN_PROMPT summaries untouched
        and merges everything older into compressed_text.
        """
        summaries: list[dict] = _read_json(SUMMARIES_PATH)
        # Keep the newest summaries as-is; replace the rest with compressed
        keep = summaries[-MAX_SUMMARIES_IN_PROMPT:]
        merged = {
            "summary": compressed_text,
            "created_at": datetime.utcnow().isoformat(),
            "type": "compressed",
        }
        _write_json(SUMMARIES_PATH, [merged] + keep)

    def get_all_summaries(self) -> list[dict]:
        """Return all summaries (used when building compression prompts)."""
        return _read_json(SUMMARIES_PATH)

    # ── Layer 3: User profile ──────────────────────────────────────────────────

    def get_profile(self) -> dict:
        """Return the full user profile."""
        return _read_json(PROFILE_PATH)

    def update_profile(self, updates: dict) -> None:
        """
        Merge new facts into the profile.
        Only call this when genuinely durable facts are discovered.
        Never stores temporary/session info.
        """
        profile: dict = _read_json(PROFILE_PATH)
        profile.update(updates)
        _write_json(PROFILE_PATH, profile)

    # ── Context Builder ────────────────────────────────────────────────────────

    def build_messages(self, current_user_message: str) -> list[BaseMessage]:
        """
        Assemble the conversation as a list of native LangChain BaseMessage objects.

        Message order (oldest → newest, matching LLM expectations):
          1. HumanMessage with profile + summaries as context preamble
          2. AIMessage acknowledging the context  (keeps roles alternating cleanly)
          3. Alternating HumanMessage / AIMessage from recent window
          4. HumanMessage with the current user message

        Why messages instead of a string?
          - The LLM's tool-calling API requires proper message objects.
          - ToolMessage responses (tool results) slot naturally into this list.
          - Future RAG docs can be prepended to the preamble HumanMessage.
          - Future multi-turn tool calls maintain correct role alternation.

        Token efficiency is preserved: only profile + 3 summaries + 10 messages
        are ever sent, regardless of how long the conversation history is.
        """
        messages: list[BaseMessage] = []

        # ── Build context preamble (profile + summaries) ───────────────────────
        preamble_parts: list[str] = []

        profile = self.get_profile()
        if profile:
            profile_str = json.dumps(profile, indent=2, ensure_ascii=False)
            preamble_parts.append(f"[USER PROFILE]\n{profile_str}")

        summaries = self.get_recent_summaries()
        if summaries:
            summary_lines = "\n\n".join(
                f"Summary ({s.get('created_at', '')[:10]}):\n{s['summary']}"
                for s in summaries
            )
            preamble_parts.append(f"[CONVERSATION HISTORY — SUMMARIES]\n{summary_lines}")

        if preamble_parts:
            preamble_text = (
                "Here is context about the user and our past conversations:\n\n"
                + "\n\n".join(preamble_parts)
            )
            # Inject as the first human turn; the AI acknowledges it briefly.
            messages.append(HumanMessage(content=preamble_text))
            messages.append(AIMessage(content="Understood. I have your profile and conversation history in mind."))

        # ── Replay recent conversation window ──────────────────────────────────
        recent = self._get_recent_window()
        for msg in recent:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))

        # ── Current user message ───────────────────────────────────────────────
        messages.append(HumanMessage(content=current_user_message))

        return messages

    # Keep the old method as an alias so nothing breaks if it's called elsewhere
    def build_context(self, current_user_message: str) -> str:
        """
        Deprecated: returns a flat string version of the context.
        Kept for backward compatibility. Prefer build_messages().
        """
        msgs = self.build_messages(current_user_message)
        return "\n".join(
            f"{'USER' if isinstance(m, HumanMessage) else 'ASSISTANT'}: {m.content}"
            for m in msgs
        )
