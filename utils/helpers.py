"""Small shared helpers for the Autonomous Cognitive Engine."""

import re


def message_text(content) -> str:
    """Flatten a message's content into plain text.

    Some models return content as a plain string, others as a list of content
    blocks like [{"type": "text", "text": "..."}]. This normalizes both.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def clean_for_display(text: str) -> str:
    """Turn raw <br> tags from the model into real Markdown line breaks."""
    return re.sub(r"<br\s*/?>", "  \n", text)