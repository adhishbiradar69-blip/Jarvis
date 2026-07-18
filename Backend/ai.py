"""
ai.py — LangChain + Groq tool-calling engine for Jarvis.

v2 changes:
  - LLM is bound to ALL_TOOLS via llm.bind_tools()
  - generate_response() runs a tool-calling loop:
      1. Ask LLM (with tools available)
      2. If LLM picks a tool → execute it → feed result back → ask LLM again
      3. If LLM replies directly → return the text
  - Memory context is sent as list[BaseMessage] (native LangChain objects)
  - Summarisation / profile update calls remain plain text (no tools needed there)

Future extension points (marked with # FUTURE):
  - RAG: inject retrieved docs into build_messages() before the loop
  - Vision: add image content blocks to the HumanMessage
  - Streaming: add streaming=True and a StreamingCallbackHandler
  - Agents: replace the tool loop with a LangChain AgentExecutor
  - Multiple LLM providers: swap ChatGroq for ChatOpenAI / ChatAnthropic etc.
  - MCP Servers: add MCP tool wrappers to ALL_TOOLS in tools/__init__.py
"""

import os
import json
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
from tools import ALL_TOOLS  # single import; new tools are added in tools/__init__.py

# ── Environment ────────────────────────────────────────────────────────────────

load_dotenv()

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY is not set. Add it to your .env file.")

# ── Model initialisation ───────────────────────────────────────────────────────

MODEL_NAME: str = "llama-3.1-8b-instant"

# Base LLM — no tools bound yet
_base_llm: ChatGroq = ChatGroq(
    model=MODEL_NAME,
    api_key=GROQ_API_KEY,
    temperature=0.7,
    max_tokens=1024,
    # FUTURE: add streaming=True + callback_manager=[StreamingHandler()] for TTS
)

# Tool-aware LLM — used for all user-facing calls
# bind_tools() tells the model which tools exist and what their signatures are
_llm = _base_llm.bind_tools(ALL_TOOLS)

# Plain LLM (no tools) — used for summarisation / profile extraction
# We don't want the model trying to call tools during memory maintenance
_plain_llm = _base_llm


# ── Tool execution ─────────────────────────────────────────────────────────────

# Map tool names → callable objects (built from ALL_TOOLS at startup)
_tool_map: dict[str, Any] = {t.name: t for t in ALL_TOOLS}


def _execute_tool_call(tool_name: str, tool_args: dict) -> str:
    """
    Look up a tool by name and invoke it with the given arguments.
    Returns a string result (or error message) to feed back to the LLM.
    """
    tool_fn = _tool_map.get(tool_name)
    if tool_fn is None:
        return f"Error: unknown tool '{tool_name}'"
    try:
        result = tool_fn.invoke(tool_args)
        return str(result)
    except Exception as exc:
        return f"Tool '{tool_name}' raised an error: {exc}"


# ── Tool-calling loop ──────────────────────────────────────────────────────────

def _run_tool_loop(messages: list[BaseMessage]) -> str:
    """
    Core agentic loop:

      while True:
        1. Send messages to LLM (with tools bound)
        2. If LLM returns tool_calls → execute each tool → append ToolMessage → continue
        3. If LLM returns a text reply → return it

    The loop handles multiple consecutive tool calls (e.g. open app, then screenshot).
    Max iterations cap prevents infinite loops.

    Args:
        messages: The full conversation so far as LangChain message objects.

    Returns:
        The final text reply from the assistant.
    """
    MAX_ITERATIONS = 8  # safety cap
    loop_messages = list(messages)  # don't mutate caller's list

    for _ in range(MAX_ITERATIONS):
        try:
            response: AIMessage = _llm.invoke(loop_messages)
        except Exception as exc:
            print(f"[ai.py] LLM call failed: {exc}")
            return "I ran into an error reaching the AI. Please try again."

        # If the LLM made tool calls, execute them and loop back
        if response.tool_calls:
            # Append the assistant's tool-call message to history
            loop_messages.append(response)

            # Execute every tool the LLM requested (may be multiple in one turn)
            for tool_call in response.tool_calls:
                result = _execute_tool_call(
                    tool_name=tool_call["name"],
                    tool_args=tool_call["args"],
                )
                print(f"[Tool] {tool_call['name']}({tool_call['args']}) → {result[:120]}")

                # Feed the result back as a ToolMessage (required by LangChain)
                loop_messages.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=tool_call["id"],
                    )
                )
            # Loop: the LLM will now see the tool results and reply
            continue

        # No tool calls → plain text reply → we're done
        return response.content.strip() if response.content else ""

    return "I reached the maximum number of steps. Please try again."


# ── Helper LLM calls (no tools) ───────────────────────────────────────────────

def _call_plain(prompt: str) -> str:
    """Call the LLM without tools for memory maintenance tasks."""
    try:
        response = _plain_llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as exc:
        print(f"[ai.py] Plain LLM call failed: {exc}")
        return ""


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
        "Merge the following summaries into a single, dense summary.\n"
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
        "Given the current user profile and the latest conversation, "
        "extract any NEW durable facts that should be remembered permanently.\n\n"
        "Rules:\n"
        "- Only include genuinely long-term information (name, language preferences, "
        "ongoing projects, permanent goals, coding style, etc.).\n"
        "- Do NOT include temporary info (today's task, what they ate, etc.).\n"
        "- If nothing new was discovered, return an empty JSON object: {}\n"
        "- Return ONLY valid JSON. No explanation, no markdown fences.\n\n"
        f"CURRENT PROFILE:\n{profile_str}\n\n"
        f"LATEST EXCHANGE:\n{recent_exchange}\n\nNEW FACTS (JSON):"
    )


# ── Memory lifecycle ───────────────────────────────────────────────────────────

def _maybe_summarize(memory: MemoryManager) -> None:
    """Summarise recent messages if the window is full; compress if needed."""
    if not memory.should_summarize():
        return
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
    """Extract new permanent facts from the latest exchange and update profile.json."""
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

    Flow:
      1. Build context as list[BaseMessage] (profile + summaries + recent + current)
      2. Run the tool-calling loop (LLM decides if tools are needed)
      3. Persist user message and assistant reply to recent.json
      4. Optionally update user profile
      5. Optionally trigger summarisation / compression
      6. Return the final text reply

    The tool-calling loop is transparent to main.py — it just receives the
    final text response regardless of how many tool calls happened internally.

    Future extension points:
      - RAG: build_messages() can accept retrieved_docs parameter
      - Vision: add image content to the HumanMessage before the loop
      - TTS: call your speech engine on the returned reply in main.py
      - Streaming: replace _run_tool_loop with a streaming equivalent
    """
    # Build native message list
    messages: list[BaseMessage] = memory.build_messages(user_message)

    # Run tool-calling loop (handles 0 or many tool calls transparently)
    reply: str = _run_tool_loop(messages)

    # Persist both sides of the exchange immediately (crash-safe)
    memory.add_message("user", user_message)
    memory.add_message("assistant", reply)

    # Background memory maintenance
    _maybe_update_profile(memory, user_message, reply)
    _maybe_summarize(memory)

    return reply
