"""
ai.py — Refactored LangChain + Groq engine for Jarvis. (v3)

Key changes from v2:
  ┌─────────────────────────────────────────────────────────────────────┐
  │ Problem → Fix                                                       │
  ├─────────────────────────────────────────────────────────────────────┤
  │ P1  Unnecessary tool calls     → tool_choice="auto" + system hint  │
  │ P2  Too many LLM calls         → max 2 calls (decide + finalise)   │
  │ P3  Profile update every turn  → heuristic filter before calling   │
  │ P4  Summarisation always runs  → only when threshold truly reached │
  │ P5  429 crashes assistant      → groq RateLimitError caught        │
  │ P6  Loop calls LLM after each  → execute ALL tools, then ONE call  │
  │ P7  Infinite tool loop         → MAX_TOOL_ROUNDS = 3               │
  │ P8  Duplicate tool execution   → executed_set dedup per request    │
  │ P9  Tool schema failures       → validated @tool signatures        │
  │ P10 Memory save timing         → persisted before any maintenance  │
  └─────────────────────────────────────────────────────────────────────┘
"""

import os
import json
import re
from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)

from memory import MemoryManager
from tools import ALL_TOOLS

# ── Environment ────────────────────────────────────────────────────────────────

load_dotenv()

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY is not set. Add it to your .env file.")

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_NAME: str = "llama-3.1-8b-instant"

# Fix P7: hard cap on how many tool-execution rounds can happen per request.
# Round = one LLM response that contains tool calls.
# After MAX_TOOL_ROUNDS we stop and ask the LLM to reply with whatever it has.
MAX_TOOL_ROUNDS: int = 3

# Words that hint a message probably contains long-term-worthy information.
# Used by the profile-update heuristic (Fix P3).
_PROFILE_HINT_WORDS: frozenset[str] = frozenset({
    "name", "called", "prefer", "favourite", "favorite", "language", "project",
    "building", "working", "goal", "always", "never", "like", "love", "hate",
    "editor", "vscode", "vim", "style", "reminder", "remember",
})

# ── Model initialisation ───────────────────────────────────────────────────────

_base_llm: ChatGroq = ChatGroq(
    model=MODEL_NAME,
    api_key=GROQ_API_KEY,
    temperature=0.7,
    max_tokens=1024,
    # FUTURE: streaming=True + callback_manager for TTS
)

# Tool-aware LLM.
# tool_choice="auto" tells Groq: use tools only when genuinely needed.
# This is the primary fix for P1 (unnecessary tool calls).
_llm = _base_llm.bind_tools(ALL_TOOLS, tool_choice="auto")

# Plain LLM used only for summarisation and compression — never for user turns.
_plain_llm = _base_llm

# Tool dispatch map built once at startup.
_tool_map: dict[str, Any] = {t.name: t for t in ALL_TOOLS}


# ── Groq error handling ────────────────────────────────────────────────────────

def _is_rate_limit(exc: Exception) -> bool:
    """Return True if the exception is a Groq 429 rate-limit error (Fix P5)."""
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "rate_limit" in msg


def _call_llm_safe(messages: list[BaseMessage], use_tools: bool = True) -> AIMessage | None:
    """
    Invoke the LLM and handle common errors gracefully (Fix P5).

    Returns the AIMessage on success, or None on unrecoverable error.
    Prints a user-friendly message instead of crashing.
    """
    llm = _llm if use_tools else _plain_llm
    try:
        return llm.invoke(messages)
    except Exception as exc:
        if _is_rate_limit(exc):
            print("\n[Jarvis] Groq rate limit reached. Please wait a few seconds and try again.\n")
        else:
            print(f"\n[Jarvis] LLM error: {exc}\n")
        return None


# ── Tool execution ─────────────────────────────────────────────────────────────

def _execute_tool_call(tool_name: str, tool_args: dict) -> str:
    """Invoke a tool by name and return its string result (Fix P9)."""
    tool_fn = _tool_map.get(tool_name)
    if tool_fn is None:
        return f"Error: unknown tool '{tool_name}'"
    try:
        result = tool_fn.invoke(tool_args)
        return str(result)
    except Exception as exc:
        return f"Tool '{tool_name}' raised an error: {exc}"


def _tool_dedup_key(tool_call: dict) -> str:
    """Build a hashable key for deduplication (Fix P8)."""
    return f"{tool_call['name']}::{json.dumps(tool_call.get('args', {}), sort_keys=True)}"


# ── Core tool-calling loop ─────────────────────────────────────────────────────

def _run_tool_loop(messages: list[BaseMessage]) -> str:
    """
    Optimised tool-calling loop (Fixes P2, P6, P7, P8).

    Algorithm (max 2 LLM calls for a tool request):

      Round 1 — Decision call:
        Send messages → LLM
        If no tool_calls → return text immediately (1 LLM call total).
        If tool_calls    → execute ALL of them in one batch (no extra LLM call).

      Round 2 — Finalisation call (only if tools were used):
        Append all ToolMessages → LLM → return text.

      Extra rounds (up to MAX_TOOL_ROUNDS):
        Only if the finalisation response itself requests more tools.
        Extremely rare; the cap (default 3) prevents infinite loops.

    Deduplication (Fix P8):
        A set tracks (tool_name, args) pairs. The same call is never repeated.
    """
    loop_messages = list(messages)           # don't mutate caller's list
    executed: set[str] = set()               # dedup tracker for this request

    for round_num in range(MAX_TOOL_ROUNDS):
        response = _call_llm_safe(loop_messages, use_tools=True)
        if response is None:
            return "I couldn't reach the AI right now. Please try again in a moment."

        # ── No tool calls → plain text reply → done (1 LLM call) ──────────────
        if not response.tool_calls:
            return response.content.strip() if response.content else ""

        # ── Tool calls present → execute the whole batch ───────────────────────
        loop_messages.append(response)  # add assistant's tool-call message

        any_executed = False
        for tool_call in response.tool_calls:
            key = _tool_dedup_key(tool_call)

            # Fix P8: skip duplicate calls silently
            if key in executed:
                print(f"[Tool] Skipping duplicate: {tool_call['name']}")
                # Still need a ToolMessage placeholder so the message sequence is valid
                loop_messages.append(ToolMessage(
                    content="(skipped: already executed this call)",
                    tool_call_id=tool_call["id"],
                ))
                continue

            executed.add(key)
            result = _execute_tool_call(tool_call["name"], tool_call.get("args", {}))
            print(f"[Tool] {tool_call['name']}({tool_call.get('args', {})}) → {result[:120]}")

            loop_messages.append(ToolMessage(
                content=result,
                tool_call_id=tool_call["id"],
            ))
            any_executed = True

        # If we hit the round cap, do one final LLM call and exit
        if round_num == MAX_TOOL_ROUNDS - 1:
            print(f"[Jarvis] Tool round cap ({MAX_TOOL_ROUNDS}) reached. Finalising.")
            final = _call_llm_safe(loop_messages, use_tools=False)
            if final is None:
                return "I ran into an error while finalising the response."
            return final.content.strip() if final.content else ""

        # Otherwise loop back (round 2 will usually be the final text reply)
        # This covers the rare case where the LLM requests a second batch of tools

    # Should never reach here, but just in case
    return "I reached the maximum number of steps. Please try again."


# ── Memory maintenance helpers ─────────────────────────────────────────────────

def _call_plain(prompt: str) -> str:
    """Call the plain (no-tool) LLM for memory tasks. Returns '' on failure."""
    response = _call_llm_safe([HumanMessage(content=prompt)], use_tools=False)
    return response.content.strip() if response and response.content else ""


def _build_summary_prompt(messages: list[dict]) -> str:
    dialog = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
    return (
        "You are a memory compression assistant.\n"
        "Summarise the following conversation for long-term storage.\n"
        "Rules:\n"
        "- Preserve: unfinished tasks, goals, user decisions, project info, important context.\n"
        "- Discard: small talk, pleasantries, repetitive exchanges.\n"
        "- Be concise. Use bullet points. No filler.\n\n"
        f"CONVERSATION:\n{dialog}\n\nSUMMARY:"
    )


def _build_compression_prompt(summaries: list[dict]) -> str:
    all_text = "\n\n".join(
        f"[{s.get('created_at', '')[:10]}] {s['summary']}" for s in summaries
    )
    return (
        "You are a memory compression assistant.\n"
        "Merge the following summaries into one dense summary.\n"
        "Rules:\n"
        "- Eliminate duplicates.\n"
        "- Preserve every unique fact, goal, project, and decision.\n"
        "- Be concise. Use bullet points.\n\n"
        f"SUMMARIES:\n{all_text}\n\nMERGED SUMMARY:"
    )


def _build_profile_update_prompt(current_profile: dict, recent_exchange: str) -> str:
    profile_str = json.dumps(current_profile, indent=2, ensure_ascii=False)
    return (
        "You are a user-profile extraction assistant.\n"
        "Extract any NEW durable facts from the exchange below.\n"
        "Rules:\n"
        "- Long-term only: name, language preferences, ongoing projects, goals, coding style.\n"
        "- Skip temporary info (today's task, casual chat, questions, greetings).\n"
        "- Return ONLY valid JSON. Empty object {} if nothing new.\n"
        "- No markdown fences, no explanation.\n\n"
        f"CURRENT PROFILE:\n{profile_str}\n\n"
        f"EXCHANGE:\n{recent_exchange}\n\nNEW FACTS (JSON):"
    )


# ── Profile-update heuristic (Fix P3) ─────────────────────────────────────────

def _exchange_may_contain_profile_info(user_message: str) -> bool:
    """
    Cheap local check: does this message plausibly contain long-term facts?

    We scan for hint words (no LLM call). If none match, we skip the
    profile-extraction LLM call entirely. This eliminates the extra call for
    greetings, math questions, tool requests, casual chat, etc.
    """
    words = set(re.findall(r"\b\w+\b", user_message.lower()))
    return bool(words & _PROFILE_HINT_WORDS)


# ── Memory lifecycle ───────────────────────────────────────────────────────────

def _maybe_summarize(memory: MemoryManager) -> None:
    """
    Summarise and optionally compress only when the threshold is actually reached.
    The threshold check is a fast local JSON read — no LLM call unless needed.
    (Fix P4)
    """
    if not memory.should_summarize():
        return  # ← most calls exit here with zero LLM usage

    print("[Jarvis] Compressing conversation memory…")
    summary = _call_plain(_build_summary_prompt(memory.get_recent_messages()))
    if summary:
        memory.append_summary(summary)

    if memory.needs_compression():
        print("[Jarvis] Merging old summaries…")
        compressed = _call_plain(_build_compression_prompt(memory.get_all_summaries()))
        if compressed:
            memory.compress_summaries(compressed)


def _maybe_update_profile(
    memory: MemoryManager,
    user_message: str,
    assistant_reply: str,
) -> None:
    """
    Update profile.json only if the heuristic suggests new facts exist.
    (Fix P3)
    """
    if not _exchange_may_contain_profile_info(user_message):
        return  # ← skip the LLM call entirely for greetings, tool calls, etc.

    exchange = f"USER: {user_message}\nASSISTANT: {assistant_reply}"
    raw = _call_plain(_build_profile_update_prompt(memory.get_profile(), exchange))
    if not raw:
        return
    try:
        clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        updates: dict[str, Any] = json.loads(clean)
        if updates:
            memory.update_profile(updates)
    except json.JSONDecodeError:
        pass


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_response(user_message: str, memory: MemoryManager) -> str:
    """
    Main entry point called from main.py.

    Typical call cost:
      Normal conversation → 1 LLM call
      Tool request        → 2 LLM calls  (decide + finalise)
      Profile update      → +1 only when hint words detected in user message
      Summarisation       → +1 only when recent.json hits SUMMARY_TRIGGER

    Flow:
      1. Build messages (profile + summaries + recent window + current)
      2. Run optimised tool loop
      3. Persist IMMEDIATELY (Fix P10) — before any maintenance
      4. Conditionally update profile (Fix P3)
      5. Conditionally summarise (Fix P4)
      6. Return reply

    Future extension points:
      - RAG: pass retrieved_docs into build_messages()
      - Vision: add image content blocks to the final HumanMessage
      - Streaming: swap _run_tool_loop for a streaming variant
      - Agents: replace _run_tool_loop with AgentExecutor
      - MCP: add MCP tool wrappers to tools/__init__.py ALL_TOOLS
    """
    messages: list[BaseMessage] = memory.build_messages(user_message)

    reply: str = _run_tool_loop(messages)

    # Fix P10: persist BOTH sides immediately — before any other work
    memory.add_message("user", user_message)
    memory.add_message("assistant", reply)

    # Conditional maintenance — each skips gracefully when not needed
    _maybe_update_profile(memory, user_message, reply)
    _maybe_summarize(memory)

    return reply
