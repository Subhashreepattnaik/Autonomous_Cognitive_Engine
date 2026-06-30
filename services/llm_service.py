"""
LLM service: single access point for all model calls, with automatic
rate limiting and failover (Gemini -> gpt-oss-120b -> llama-3.3-70b).

Every model call in the project goes through get_llm() / invoke_llm(), so the
shared rate limiter and automatic model failover apply everywhere from one place.
"""

import os
from langchain_core.rate_limiters import InMemoryRateLimiter

from config import settings

# --- Shared rate limiter (paces every request to respect free-tier limits) ---
# Gemini's free tier allows more throughput than Groq, so 2 req/s is safe and
# much faster than the old 0.5. Lower this if you start seeing rate-limit errors.
_rate_limiter = InMemoryRateLimiter(
    requests_per_second=2,
    check_every_n_seconds=0.1,
    max_bucket_size=3,
)

# --- Failover chain: (provider, model_name). Tried in order on a 429/quota error.
#     Gemini is the main model; Groq models are fallbacks. gpt-oss-20b removed. ---
_MODELS = [
    ("groq",   "llama-3.3-70b-versatile"),  # main — fast, strong writing, no thinking
    ("google", "gemini-2.0-flash"),         # fallback 1 — fast, generous limits
    ("groq",   "openai/gpt-oss-120b"),      # fallback 2 — capable, last resort
]
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


def _build_gemini(model_name: str, temperature):
    """Construct a Gemini model. Disable 'thinking' for speed (gemini-2.5-flash
    is a reasoning model; we don't need internal thinking for these tasks)."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    base = dict(
        model=model_name,
        temperature=temperature,
        max_output_tokens=4000,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        rate_limiter=_rate_limiter,
    )
    # Try to disable thinking for speed. Library versions differ on how this is
    # passed, so fall back gracefully if the argument isn't supported.
    try:
        return ChatGoogleGenerativeAI(**base, thinking_budget=0)
    except TypeError:
        try:
            return ChatGoogleGenerativeAI(
                **base, model_kwargs={"thinking_config": {"thinking_budget": 0}}
            )
        except TypeError:
            return ChatGoogleGenerativeAI(**base)  # no thinking control available


def _build(provider: str, model_name: str, temperature):
    if provider == "google":
        return _build_gemini(model_name, temperature)
    return _build_groq(model_name, temperature)


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
            "resource_exhausted",   # Gemini's quota error wording
            "resource exhausted",
        )
    )


def get_llm(temperature=None):
    """Return the currently active model (advances automatically on a 429)."""
    provider, model_name = _MODELS[_active_idx]
    print(f"[MODEL] {provider}/{model_name}")   # shows which model each call uses
    return _build(provider, model_name, _resolve_temp(temperature))


def call_with_failover(make_runnable, payload):
    """
    Invoke a freshly-built runnable; on a 429 / quota error, advance to the next
    model and retry. `make_runnable` is a zero-arg function that builds the
    runnable via get_llm() so it picks up the (possibly advanced) current model.
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
                    f"{_MODELS[_active_idx][1]}"
                )
                continue
            raise


def invoke_llm(prompt, temperature=None):
    """A direct LLM call with automatic model failover."""
    return call_with_failover(lambda: get_llm(temperature), prompt)