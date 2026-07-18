"""
tools/app_tools.py — Application launching tools.

Supports Windows (subprocess / start), macOS (open), and Linux (xdg-open).
The LLM calls these when the user asks to open an application.
"""

import subprocess
import sys
from langchain_core.tools import tool


def _launch(command: list[str]) -> str:
    """
    Internal helper: run a subprocess and return a status string.
    Never raises — always returns a human-readable result.
    """
    try:
        subprocess.Popen(command, shell=(sys.platform == "win32"))
        return "success"
    except FileNotFoundError:
        return f"Application not found: {command}"
    except Exception as exc:
        return f"Error launching application: {exc}"


@tool
def open_application(app_name: str) -> str:
    """
    Open any installed application by name.
    Use this when the user asks to open an app that isn't covered by a specific tool.

    Args:
        app_name: The name of the application to open (e.g. 'Notepad', 'Calculator').
    """
    try:
        if sys.platform == "win32":
            # shell=True + string command is the correct way for 'start' on Windows
            subprocess.Popen(f'start "" "{app_name}"', shell=True)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-a", app_name])
        else:
            subprocess.Popen(["xdg-open", app_name])
        return f"Opened {app_name} successfully."
    except Exception as exc:
        return f"Could not open '{app_name}': {exc}"


@tool
def open_vscode(folder_path: str = "") -> str:
    """
    Open Visual Studio Code, optionally at a specific folder or file path.

    Args:
        folder_path: Optional path to a folder or file to open in VS Code.
                     Leave empty to just launch VS Code.
    """
    try:
        cmd = ["code"]
        if folder_path:
            cmd.append(folder_path)
        subprocess.Popen(cmd)
        target = f" at '{folder_path}'" if folder_path else ""
        return f"Opened Visual Studio Code{target}."
    except FileNotFoundError:
        return (
            "VS Code ('code' command) not found. "
            "Make sure VS Code is installed and added to PATH."
        )
    except Exception as exc:
        return f"Could not open VS Code: {exc}"


@tool
def open_chrome(url: str = "") -> str:
    """
    Open Google Chrome, optionally navigating to a URL.

    Args:
        url: Optional URL to open (e.g. 'https://google.com').
             Leave empty to open a new Chrome window.
    """
    try:
        if sys.platform == "win32":
            cmd = f'start "" chrome "{url}"' if url else 'start "" chrome'
            subprocess.Popen(cmd, shell=True)
        elif sys.platform == "darwin":
            cmd = ["open", "-a", "Google Chrome", url] if url else ["open", "-a", "Google Chrome"]
            subprocess.Popen(cmd)
        else:
            cmd = ["google-chrome", url] if url else ["google-chrome"]
            subprocess.Popen(cmd)
        target = f" to {url}" if url else ""
        return f"Opened Chrome{target}."
    except FileNotFoundError:
        return "Chrome not found. Make sure Google Chrome is installed."
    except Exception as exc:
        return f"Could not open Chrome: {exc}"


@tool
def open_spotify() -> str:
    """
    Open the Spotify desktop application.
    """
    try:
        if sys.platform == "win32":
            subprocess.Popen('start "" spotify', shell=True)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-a", "Spotify"])
        else:
            subprocess.Popen(["spotify"])
        return "Opened Spotify."
    except FileNotFoundError:
        return "Spotify not found. Make sure the Spotify desktop app is installed."
    except Exception as exc:
        return f"Could not open Spotify: {exc}"
