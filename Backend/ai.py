"""
ai.py — LangChain + Groq integration for the Jarvis backend.

Responsibilities:
  - Initialise the Groq LLM via LangChain
  - Generate responses using memory context
  - Trigger summarisation / compression / profile updates when needed

All LLM calls live here or in memory.py (for summarisation helpers called from here).
No LLM logic belongs in main.py.

Designed for easy future expansion:
  - Tool/function calling: add a tools= list to the model invocation
  - RAG: inject retrieved docs into build_context() before calling generate_response()
  - Vision: pass image content blocks alongside text
  - Agents: swap HumanMessage chain for an AgentExecutor
  - MCP / multi-agent: add orchestration layer above generate_response()
"""

import os
import json
from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from memory import MemoryManager

# ── Env ────────────────────────────────────────────────────────────────────────

load_dotenv()

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY is not set. Add it to your .env file."
    )

# ── Model ──────────────────────────────────────────────────────────────────────

MODEL_NAME: str = "llama-3.1-8b-instant"

# Initialised once at import time; reused for every call (cheap, stateless object)
_llm: ChatGroq = ChatGroq(
    model=MODEL_NAME,
    api_key=GROQ_API_KEY,
    temperature=0.7,
    max_tokens=1024,
    # Future: add streaming=True and a callback handler when you add TTS/streaming UI
)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _call_llm(prompt: str) -> str:
    """
    Send a single-turn prompt to the model and return the text response.
    All Groq calls funnel through here so error handling is centralised.
    """
    try:
        response = _llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as exc:
        # Never crash the assistant on a single failed API call
        print(f"[ai.py] LLM call failed: {exc}")
        return "I encountered an error reaching the AI. Please try again."


def _build_summary_prompt(messages: list[dict]) -> str:
    """
    Build a prompt that asks the model to summarise a batch of messages.
    The output is stored in summaries.json — keep it dense and factual.
    """
    dialog = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages
    )
    return (
        "You are a memory compression assistant.\n"
        "Summarise the following conversation for long-term storage.\n"
        "Rules:\n"
        "- Preserve: unfinished tasks, goals, user decisions, project info, important context.\n"
        "- Discard: small talk, pleasantries, repetitive exchanges.\n"
        "- Be concise. Use bullet points. No filler.\n\n"
        f"CONVERSATION:\n{dialog}\n\n"
        "SUMMARY:"
    )


def _build_compression_prompt(summaries: list[dict]) -> str:
    """
    Build a prompt that merges multiple summaries into one compressed summary.
    """
    all_text = "\n\n".join(
        f"[{s.get('created_at', '')[:10]}] {s['summary']}" for s in summaries
    )
    return (
        "You are a memory compression assistant.\n"
        "Merge the following summaries into a single, dense summary.\n"
        "Rules:\n"
        "- Eliminate duplicates.\n"
        "- Preserve every unique fact, goal, project, and decision.\n"
        "- Be concise. Use bullet points.\n\n"
        f"SUMMARIES:\n{all_text}\n\n"
        "MERGED SUMMARY:"
    )


def _build_profile_update_prompt(
    current_profile: dict,
    recent_exchange: str,
) -> str:
    """
    Ask the model to extract any new durable facts from the latest exchange
    and return them as a JSON patch to merge into profile.json.
    """
    profile_str = json.dumps(current_profile, indent=2, ensure_ascii=False)
    return (
        "You are a user-profile extraction assistant.\n"
        "Given the current user profile and the latest conversation, "
        "extract any NEW durable facts that should be remembered permanently.\n\n"
        "Rules:\n"
        "- Only include genuinely long-term information (name, language preferences, "
        "ongoing projects, permanent goals, coding style, etc.).\n"
        "- Do NOT include temporary info (today's task, what they ate, etc.).\n"
        "- If nothing new was discovered, return an empty JSON object: {}\n"
        "- Return ONLY valid JSON. No explanation, no markdown fences.\n\n"
        f"CURRENT PROFILE:\n{profile_str}\n\n"
        f"LATEST EXCHANGE:\n{recent_exchange}\n\n"
        "NEW FACTS (JSON):"
    )


# ── Memory lifecycle ───────────────────────────────────────────────────────────

def _maybe_summarize(memory: MemoryManager) -> None:
    """
    If recent.json has grown large enough, summarise and clear it.
    Optionally compress summaries if there are too many.
    """
    if not memory.should_summarize():
        return

    messages = memory.get_recent_messages()
    print("[Jarvis] Compressing conversation memory…")

    summary = _call_llm(_build_summary_prompt(messages))
    memory.append_summary(summary)  # also clears recent.json

    # If we now have too many summaries, merge the old ones
    if memory.needs_compression():
        print("[Jarvis] Merging old summaries…")
        all_summaries = memory.get_all_summaries()
        compressed = _call_llm(_build_compression_prompt(all_summaries))
        memory.compress_summaries(compressed)


def _maybe_update_profile(
    memory: MemoryManager,
    user_message: str,
    assistant_reply: str,
) -> None:
    """
    After each exchange, optionally extract new profile facts.
    We only call the LLM here — no update occurs if the model returns {}.
    """
    exchange = f"USER: {user_message}\nASSISTANT: {assistant_reply}"
    current_profile = memory.get_profile()
    prompt = _build_profile_update_prompt(current_profile, exchange)

    raw = _call_llm(prompt)

    try:
        # Strip any accidental markdown fences before parsing
        clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        updates: dict[str, Any] = json.loads(clean)
        if updates:
            memory.update_profile(updates)
    except json.JSONDecodeError:
        # Model returned something non-JSON — silently ignore
        pass


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_response(user_message: str, memory: MemoryManager) -> str:
    """
    Main entry point called from main.py.

    Flow:
      1. Build context (profile + summaries + recent window + current message)
      2. Call the LLM
      3. Save the user message and assistant reply to recent.json
      4. Optionally update the user profile
      5. Optionally trigger summarisation / compression
      6. Return the reply string

    Future extension points:
      - Step 1: inject RAG-retrieved documents into context
      - Step 2: add tools/agents via langchain AgentExecutor
      - Step 4: add TTS call here
      - After step 6: route response to speech output
    """
    # Build the full context string
    context = memory.build_context(user_message)

    # Single LLM call with the assembled context
    reply = _call_llm(context)

    # Persist both sides of the exchange immediately
    memory.add_message("user", user_message)
    memory.add_message("assistant", reply)

    # Background memory maintenance (runs synchronously; fast in practice)
    _maybe_update_profile(memory, user_message, reply)
    _maybe_summarize(memory)

    return reply
