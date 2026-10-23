"""Pytest configuration and global fixtures for ARGUS test suite."""

import pytest
from app.core.pool import reset_llm_pool


@pytest.fixture(autouse=True)
def reset_pool_between_tests():
    """Ensure every test executes with a clean, isolated LLM pool state."""
    reset_llm_pool()
    yield
    reset_llm_pool()
