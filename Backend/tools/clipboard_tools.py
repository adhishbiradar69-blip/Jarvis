"""
tools/clipboard_tools.py — Clipboard tools for Jarvis.
"""

from langchain_core.tools import tool


@tool
def copy_to_clipboard(text: str) -> str:
    """
    Copy the given text to the system clipboard.

    Args:
        text: The text to copy to the clipboard.
    """
    try:
        import pyperclip  # pip install pyperclip
        pyperclip.copy(text)
        preview = text[:60] + "..." if len(text) > 60 else text
        return f"Copied to clipboard: '{preview}'"
    except ImportError:
        return "pyperclip is not installed. Run: pip install pyperclip"
    except Exception as exc:
        return f"Could not copy to clipboard: {exc}"
