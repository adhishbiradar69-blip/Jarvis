"""
tools/utility_tools.py — General-purpose utility tools for Jarvis.

Covers: screenshot, current date, current time.
"""

import os
from datetime import datetime
from langchain_core.tools import tool


@tool
def take_screenshot(save_path: str = "") -> str:
    """
    Take a screenshot of the entire screen and save it to disk.

    Args:
        save_path: Optional full path where the screenshot should be saved.
                   Defaults to the Desktop with a timestamped filename.
    """
    try:
        import PIL.ImageGrab  # pip install Pillow

        if not save_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            os.makedirs(desktop, exist_ok=True)
            save_path = os.path.join(desktop, f"screenshot_{timestamp}.png")

        img = PIL.ImageGrab.grab()
        img.save(save_path)
        return f"Screenshot saved to: {save_path}"
    except ImportError:
        return "Pillow is not installed. Run: pip install Pillow"
    except Exception as exc:
        return f"Could not take screenshot: {exc}"


@tool
def get_current_date() -> str:
    """
    Return today's date in a human-readable format.
    Use this when the user asks what day or date it is.
    """
    return datetime.now().strftime("%A, %B %d, %Y")


@tool
def get_current_time() -> str:
    """
    Return the current local time.
    Use this when the user asks what time it is.
    """
    return datetime.now().strftime("%I:%M %p")
