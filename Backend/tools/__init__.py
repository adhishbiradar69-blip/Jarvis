"""
tools/__init__.py — Central tool registry for Jarvis.

To add new tools in the future:
  1. Create a new file in this folder (e.g. web_tools.py)
  2. Decorate your functions with @tool
  3. Import them in the ALL_TOOLS list below

The AI engine (ai.py) imports ALL_TOOLS and binds them to the LLM automatically.
No other file needs to change.
"""

from tools.app_tools      import open_application, open_vscode, open_chrome, open_spotify
from tools.file_tools     import (
    open_file, read_text_file, create_folder, list_directory,
    search_files, rename_file, delete_file, move_file, copy_file,
    read_pdf, read_docx,
)
from tools.system_tools   import (
    set_volume, set_brightness, lock_windows,
    shutdown_computer, restart_computer, sleep_computer,
    get_battery_info, get_wifi_info,
)
from tools.utility_tools  import take_screenshot, get_current_date, get_current_time
from tools.clipboard_tools import copy_to_clipboard

# ── Master tool list ───────────────────────────────────────────────────────────
# ai.py calls:  llm.bind_tools(ALL_TOOLS)
# Add imports above and append here — nothing else needs changing.

ALL_TOOLS: list = [
    # Applications
    open_application,
    open_vscode,
    open_chrome,
    open_spotify,

    # Files
    open_file,
    read_text_file,
    create_folder,
    list_directory,
    search_files,
    rename_file,
    delete_file,
    move_file,
    copy_file,
    read_pdf,
    read_docx,

    # System
    set_volume,
    set_brightness,
    lock_windows,
    shutdown_computer,
    restart_computer,
    sleep_computer,
    get_battery_info,
    get_wifi_info,

    # Utility
    take_screenshot,
    get_current_date,
    get_current_time,

    # Clipboard
    copy_to_clipboard,
]
