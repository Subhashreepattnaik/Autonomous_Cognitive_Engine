"""Small shared helpers for the Autonomous Cognitive Engine."""


def message_text(content) -> str:
    """Flatten a message's content into plain text.

    Some models (e.g. Gemini) return a message's content as a plain string,
    while others return a list of content blocks like
    [{"type": "text", "text": "..."}]. This normalizes both to a string so
    the rest of the app never has to care which shape it got.

    Args:
        content: The `.content` of a message (str or list of blocks).

    Returns:
        The text as a single string.
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