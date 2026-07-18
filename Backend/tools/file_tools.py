"""
tools/file_tools.py — File system tools for Jarvis.

Covers: open, read, create, list, search, rename, delete (with confirmation),
        move, copy, read PDF, read DOCX.

IMPORTANT — delete_file:
  The LLM must ask the user "Are you sure?" before calling delete_file.
  The tool itself also checks for a confirmed=True flag as a safety net.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from langchain_core.tools import tool


# ── Helpers ────────────────────────────────────────────────────────────────────

def _expand(path: str) -> str:
    """Expand ~ and environment variables in a path string."""
    return str(Path(path).expanduser().resolve())


# ── Phase 1: Basic file tools ──────────────────────────────────────────────────

@tool
def open_file(file_path: str) -> str:
    """
    Open a file with the system's default application.

    Args:
        file_path: Absolute or relative path to the file to open.
    """
    try:
        path = _expand(file_path)
        if not os.path.exists(path):
            return f"File not found: {path}"
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
        return f"Opened file: {path}"
    except Exception as exc:
        return f"Could not open file: {exc}"


@tool
def read_text_file(file_path: str) -> str:
    """
    Read and return the contents of a plain text file (.txt, .md, .py, .json, etc.).

    Args:
        file_path: Path to the text file.
    """
    try:
        path = _expand(file_path)
        if not os.path.exists(path):
            return f"File not found: {path}"
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        # Truncate very large files to avoid flooding the context window
        if len(content) > 8000:
            content = content[:8000] + "\n\n[... file truncated at 8000 characters ...]"
        return content
    except Exception as exc:
        return f"Could not read file: {exc}"


@tool
def create_folder(folder_path: str) -> str:
    """
    Create a new folder (and any missing parent folders).

    Args:
        folder_path: Path of the folder to create.
    """
    try:
        path = _expand(folder_path)
        os.makedirs(path, exist_ok=True)
        return f"Folder created: {path}"
    except Exception as exc:
        return f"Could not create folder: {exc}"


@tool
def list_directory(folder_path: str = ".") -> str:
    """
    List the files and folders inside a directory.

    Args:
        folder_path: Path to the directory. Defaults to the current directory.
    """
    try:
        path = _expand(folder_path)
        if not os.path.exists(path):
            return f"Directory not found: {path}"
        entries = sorted(os.listdir(path))
        if not entries:
            return f"The directory '{path}' is empty."
        lines = []
        for name in entries:
            full = os.path.join(path, name)
            kind = "📁" if os.path.isdir(full) else "📄"
            lines.append(f"{kind} {name}")
        return f"Contents of {path}:\n" + "\n".join(lines)
    except Exception as exc:
        return f"Could not list directory: {exc}"


# ── Phase 3: Advanced file tools ───────────────────────────────────────────────

@tool
def search_files(folder_path: str, pattern: str) -> str:
    """
    Recursively search for files matching a name pattern inside a folder.

    Args:
        folder_path: Root directory to search from.
        pattern: Filename pattern to match (e.g. '*.py', 'report*', 'notes.txt').
    """
    try:
        root = Path(_expand(folder_path))
        if not root.exists():
            return f"Directory not found: {root}"
        matches = list(root.rglob(pattern))
        if not matches:
            return f"No files matching '{pattern}' found in {root}."
        result = "\n".join(str(m) for m in matches[:50])  # cap results
        suffix = f"\n... and {len(matches) - 50} more" if len(matches) > 50 else ""
        return f"Found {len(matches)} match(es):\n{result}{suffix}"
    except Exception as exc:
        return f"Search failed: {exc}"


@tool
def rename_file(file_path: str, new_name: str) -> str:
    """
    Rename a file or folder. The new_name should be just the filename, not a full path.

    Args:
        file_path: Path to the file or folder to rename.
        new_name: New filename (e.g. 'report_v2.txt').
    """
    try:
        src = Path(_expand(file_path))
        if not src.exists():
            return f"Not found: {src}"
        dst = src.parent / new_name
        src.rename(dst)
        return f"Renamed '{src.name}' → '{new_name}' in {src.parent}"
    except Exception as exc:
        return f"Could not rename: {exc}"


@tool
def delete_file(file_path: str, confirmed: bool = False) -> str:
    """
    Delete a file or empty folder. The user MUST confirm before this runs.
    If confirmed is False, return a confirmation request instead of deleting.

    Args:
        file_path: Path to the file or folder to delete.
        confirmed: Must be True for deletion to proceed. Default is False (safe).
    """
    path = Path(_expand(file_path))
    if not confirmed:
        # The LLM should present this message to the user and wait for "yes"
        return (
            f"⚠️ Are you sure you want to delete '{path}'? "
            "This cannot be undone. Reply 'yes, delete it' to confirm."
        )
    try:
        if not path.exists():
            return f"Not found: {path}"
        if path.is_dir():
            shutil.rmtree(path)
            return f"Deleted folder: {path}"
        else:
            path.unlink()
            return f"Deleted file: {path}"
    except Exception as exc:
        return f"Could not delete: {exc}"


@tool
def move_file(source_path: str, destination_path: str) -> str:
    """
    Move a file or folder to a new location.

    Args:
        source_path: Current path of the file or folder.
        destination_path: Target path or directory.
    """
    try:
        src = Path(_expand(source_path))
        dst = Path(_expand(destination_path))
        if not src.exists():
            return f"Source not found: {src}"
        shutil.move(str(src), str(dst))
        return f"Moved '{src}' → '{dst}'"
    except Exception as exc:
        return f"Could not move: {exc}"


@tool
def copy_file(source_path: str, destination_path: str) -> str:
    """
    Copy a file or folder to a new location.

    Args:
        source_path: Path of the file or folder to copy.
        destination_path: Target path or directory.
    """
    try:
        src = Path(_expand(source_path))
        dst = Path(_expand(destination_path))
        if not src.exists():
            return f"Source not found: {src}"
        if src.is_dir():
            shutil.copytree(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))
        return f"Copied '{src}' → '{dst}'"
    except Exception as exc:
        return f"Could not copy: {exc}"


@tool
def read_pdf(file_path: str) -> str:
    """
    Extract and return the text content from a PDF file.

    Args:
        file_path: Path to the PDF file.
    """
    try:
        import pypdf  # pip install pypdf
        path = _expand(file_path)
        if not os.path.exists(path):
            return f"File not found: {path}"
        reader = pypdf.PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(pages).strip()
        if not text:
            return "No readable text found in the PDF (it may be scanned/image-based)."
        if len(text) > 8000:
            text = text[:8000] + "\n\n[... PDF truncated at 8000 characters ...]"
        return text
    except ImportError:
        return "pypdf is not installed. Run: pip install pypdf"
    except Exception as exc:
        return f"Could not read PDF: {exc}"


@tool
def read_docx(file_path: str) -> str:
    """
    Extract and return the text content from a Microsoft Word (.docx) file.

    Args:
        file_path: Path to the .docx file.
    """
    try:
        import docx  # pip install python-docx
        path = _expand(file_path)
        if not os.path.exists(path):
            return f"File not found: {path}"
        doc = docx.Document(path)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        if not text:
            return "No readable text found in the document."
        if len(text) > 8000:
            text = text[:8000] + "\n\n[... document truncated at 8000 characters ...]"
        return text
    except ImportError:
        return "python-docx is not installed. Run: pip install python-docx"
    except Exception as exc:
        return f"Could not read DOCX: {exc}"
