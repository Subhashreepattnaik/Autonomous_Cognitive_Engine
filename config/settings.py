"""
Central configuration for the Autonomous Cognitive Engine.
Single source of truth for:
  - secret API keys (loaded from the .env file)
  - model names and tuning constants
Every other module imports its configuration from here, so changing a key
name or a model means editing exactly one place.
"""
import os
from dotenv import load_dotenv
# Read the .env file and load its KEY=value pairs into the environment.
# After this runs, os.getenv("GOOGLE_API_KEY") returns the value from .env.
load_dotenv()
# ---------------------------------------------------------------------------
# Secret API keys (read from the environment — never hard-coded here)
# ---------------------------------------------------------------------------
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY: str | None = os.getenv("TAVILY_API_KEY")
# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
# Which LLM provider to use: "groq" (generous free RPD) or "gemini".
LLM_PROVIDER: str = "groq"
# Groq model — strong at the tool-calling our agents need. 1,000 requests/day
# free. Alternative with more daily token budget: "openai/gpt-oss-120b".
GROQ_MODEL: str = "openai/gpt-oss-120b"
#GROQ_MODEL: str = "Llama-3.3-70b-versatile"
#GROQ_MODEL: str = "openai/gpt-oss-20b"
# Gemini model (kept for easy switch-back if you get a paid key later).
GEMINI_MODEL: str = "gemini-3.5-flash"
LLM_TEMPERATURE: float = 0.3
MAX_AGENT_STEPS: int = 50

# Groq's free tier caps tokens-per-minute (~12K for the 70B model), so we pace
# requests conservatively (~4/min) to avoid token-based 429s. The request-rate
# limit (30/min) is generous; tokens are the real ceiling.
LLM_REQUESTS_PER_SECOND: float = 0.066
# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_settings() -> None:
    """Ensure every required secret is present.
    Raises:
        EnvironmentError: if any required key is missing, naming exactly
            which ones so setup problems are obvious instead of mysterious.
    """
    required = {
        "TAVILY_API_KEY": TAVILY_API_KEY,
    }
    if LLM_PROVIDER == "groq":
        required["GROQ_API_KEY"] = GROQ_API_KEY
    else:
        required["GOOGLE_API_KEY"] = GOOGLE_API_KEY

    missing = [name for name, value in required.items() if not value]

    if missing:
        raise EnvironmentError(
            "Missing required API key(s): "
            + ", ".join(missing)
            + ". Add them to your .env file (see .env.example)."
        )