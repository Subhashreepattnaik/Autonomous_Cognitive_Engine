"""
Virtual file system tools — the agent's context-offloading memory.

Instead of holding large findings in the LLM's limited context window, the
agent saves them to a virtual file system (a dict in the graph state) and
reads them back on demand.

  ls          list saved files
  read_file   read one file's contents
  write_file  create or overwrite a file
  edit_file   replace a snippet inside an existing file

Read-only tools (ls, read_file) return a plain string and the tool runner
wraps it in a ToolMessage automatically. State-changing tools (write_file,
edit_file) return a Command and include the ToolMessage themselves.
"""

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command


@tool
def ls(runtime: ToolRuntime) -> str:
    """List the names of all files saved in the virtual file system."""
    files = runtime.state.get("virtual_files", {})
    if not files:
        return "The virtual file system is empty."
    names = "\n".join(f"  - {name}" for name in files)
    return f"Saved files ({len(files)}):\n{names}"


@tool
def read_file(filename: str, runtime: ToolRuntime) -> str:
    """Read and return the full contents of a saved file.

    Args:
        filename: The name of the file to read.
    """
    files = runtime.state.get("virtual_files", {})
    if filename not in files:
        available = ", ".join(files) if files else "none"
        return f"Error: file '{filename}' not found. Available files: {available}."
    return files[filename]


@tool
def write_file(filename: str, content: str, runtime: ToolRuntime) -> Command:
    """Create a new file or overwrite an existing one with the given content.

    Use this to save findings, summaries, or drafts so they persist without
    filling up the context window.

    Args:
        filename: The name to save the file under.
        content: The text to write into the file.
    """
    # Copy the current files dict (never mutate state in place), then update.
    files = dict(runtime.state.get("virtual_files", {}))
    files[filename] = content

    message = f"Saved '{filename}' ({len(content)} characters)."
    return Command(
        update={
            "virtual_files": files,
            "messages": [
                ToolMessage(content=message, tool_call_id=runtime.tool_call_id)
            ],
        }
    )


@tool
def edit_file(
    filename: str,
    old_string: str,
    new_string: str,
    runtime: ToolRuntime,
) -> Command:
    """Replace the first occurrence of old_string with new_string in a file.

    Use this for small, targeted edits instead of rewriting a whole file.

    Args:
        filename: The file to edit.
        old_string: The exact text to find.
        new_string: The text to replace it with.
    """
    files = dict(runtime.state.get("virtual_files", {}))

    if filename not in files:
        available = ", ".join(files) if files else "none"
        note = f"Error: file '{filename}' not found. Available files: {available}."
        return Command(
            update={
                "messages": [
                    ToolMessage(content=note, tool_call_id=runtime.tool_call_id)
                ]
            }
        )

    if old_string not in files[filename]:
        note = f"Error: text to replace was not found in '{filename}'."
        return Command(
            update={
                "messages": [
                    ToolMessage(content=note, tool_call_id=runtime.tool_call_id)
                ]
            }
        )

    files[filename] = files[filename].replace(old_string, new_string, 1)
    message = f"Edited '{filename}' successfully."
    return Command(
        update={
            "virtual_files": files,
            "messages": [
                ToolMessage(content=message, tool_call_id=runtime.tool_call_id)
            ],
        }
    )