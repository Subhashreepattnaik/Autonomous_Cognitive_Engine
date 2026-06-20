"""
LLM service: single access point for all model calls, with automatic
rate limiting and 429 failover (120B -> 20B -> 70B).
"""

from langchain_core.rate_limiters import InMemoryRateLimiter

from config import settings

# --- Shared rate limiter (paces every request to respect free-tier limits) ---
_rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,     # ~1 request every 2s; adjust if needed
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)

# --- Model failover chain: tried in order when a 429 / quota error occurs ---
_MODELS = [
    settings.GROQ_MODEL,            # main (openai/gpt-oss-120b from settings)
    "openai/gpt-oss-20b",          # fallback 1: separate fresh quota bucket
    "llama-3.3-70b-versatile",     # fallback 2: last resort
]
# de-duplicate while preserving order (in case GROQ_MODEL already appears)
_MODELS = list(dict.fromkeys(_MODELS))
_active_idx = 0


def _resolve_temp(temperature):
    return temperature if temperature is not None else settings.LLM_TEMPERATURE


def _build_groq(model_name: str, temperature):
    """Construct a ChatGroq model. reasoning_effort only on gpt-oss models."""
    from langchain_groq import ChatGroq

    kwargs = dict(
        model=model_name,
        temperature=temperature,
        max_tokens=4000,
        max_retries=2,               # fail over sooner instead of retrying a dead model
        rate_limiter=_rate_limiter,
    )
    if model_name.startswith("openai/gpt-oss"):
        kwargs["reasoning_effort"] = "low"
    return ChatGroq(**kwargs)


def _is_rate_limit(exc) -> bool:
    s = str(exc).lower()
    return any(
        k in s
        for k in (
            "429",
            "rate limit",
            "rate_limit",
            "too many requests",
            "tokens per day",
            "quota",
        )
    )


def get_llm(temperature=None):
    """Return the currently active model (advances automatically on 429)."""
    return _build_groq(_MODELS[_active_idx], _resolve_temp(temperature))


def call_with_failover(make_runnable, payload):
    """
    Invoke a freshly-built runnable; on a 429, advance to the next model and
    retry. `make_runnable` is a zero-arg function that builds the runnable
    using get_llm() so it picks up the (possibly advanced) current model.
    """
    global _active_idx
    while True:
        runnable = make_runnable()
        try:
            return runnable.invoke(payload)
        except Exception as exc:
            if _is_rate_limit(exc) and _active_idx < len(_MODELS) - 1:
                _active_idx += 1
                print(
                    f"[failover] rate limit hit -> switching to "
                    f"{_MODELS[_active_idx]}"
                )
                continue
            raise


def invoke_llm(prompt, temperature=None):
    """A direct LLM call with automatic model failover."""
    return call_with_failover(lambda: get_llm(temperature), prompt)