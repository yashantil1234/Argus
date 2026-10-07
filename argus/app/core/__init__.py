"""Core module: LLM abstraction layer."""

from app.core.llm import (
    LLMProvider,
    BaseLLMClient,
    OllamaClient,
    OpenRouterClient,
    get_llm_client,
    ask_llm,
)

__all__ = [
    "LLMProvider",
    "BaseLLMClient",
    "OllamaClient",
    "OpenRouterClient",
    "get_llm_client",
    "ask_llm",
]
