"""
LLM service — creates the rate-limited language model that powers every agent.

Provider is chosen in config.settings (LLM_PROVIDER). Because all agents route
through get_llm(), switching providers or models is a config change — no agent
code is touched. One shared rate limiter paces the whole system.
"""

from langchain_core.rate_limiters import InMemoryRateLimiter

from config import settings

# ONE shared limiter for every model this service creates, so all agents
# together stay under the provider's free-tier request rate.
_rate_limiter = InMemoryRateLimiter(
    requests_per_second=settings.LLM_REQUESTS_PER_SECOND,
    check_every_n_seconds=0.5,
    max_bucket_size=1,
)


def get_llm(temperature: float | None = None, model: str | None = None):
    """Create and return a configured, rate-limited chat model.

    The provider (Groq or Gemini) comes from settings.LLM_PROVIDER.

    Args:
        temperature: Optional randomness override (0.0-1.0).
        model: Optional model-name override.

    Returns:
        A ready-to-use, rate-limited chat model.
    """
    temp = temperature if temperature is not None else settings.LLM_TEMPERATURE

    if settings.LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model or settings.GROQ_MODEL,
            temperature=temp,
            max_retries=5,
            rate_limiter=_rate_limiter,
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model or settings.GEMINI_MODEL,
        temperature=temp,
        google_api_key=settings.GOOGLE_API_KEY,
        max_retries=5,
        rate_limiter=_rate_limiter,
    )


if __name__ == "__main__":
    settings.validate_settings()
    llm = get_llm()
    print(llm.invoke("In one short sentence, what is an AI agent?").content)