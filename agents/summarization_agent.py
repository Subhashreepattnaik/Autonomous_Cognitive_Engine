"""
Summarization agent — condenses long findings into a tight, structured
summary. It needs no tools: given text, it returns a summary. A deliberate
reminder that not every specialist needs a full tool-using loop.
"""
from langchain.agents import create_agent
from services.llm_service import get_llm

SUMMARIZATION_SYSTEM_PROMPT = """You are an expert summarizer.
You will be given research findings or other text. Produce a clear, concise
summary that preserves the key facts, figures, and source references.
Rules:
- Keep all important facts and any cited source URLs.
- Remove repetition and filler.
- Use short paragraphs or bullet points for readability.
- Do not add information that is not in the provided text."""

def build_summarization_agent():
    """Create and return the compiled summarization agent (no tools)."""
    return create_agent(
        model=get_llm(temperature=0.2),
        tools=[],
        system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
        name="summarization_agent",
    )