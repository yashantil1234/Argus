"""Provenance and deterministic quote verification layer for ARGUS.

Ensures that evidence quotes extracted by an LLM actually exist verbatim
in the source page text, preventing hallucinated citations.

Deterministic Metric:
    Quote Fidelity = valid_verbatim_quotes / total_extracted_quotes
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Tuple


def compute_content_hash(text: str) -> str:
    """Compute deterministic SHA-256 hex digest of source text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    """Normalize whitespace, unicode punctuation, and casing for robust matching.

    Replaces curly quotes with straight quotes, normalizes dashes,
    collapses runs of whitespace, and trims.
    """
    if not text:
        return ""
    # Unicode NFKC normalization
    normalized = unicodedata.normalize("NFKC", text)

    # Normalize various quote characters to standard ASCII
    normalized = re.sub(r'[\u2018\u2019\u201A\u201B`]', "'", normalized)
    normalized = re.sub(r'[\u201C\u201D\u201E\u201F]', '"', normalized)
    # Normalize dashes
    normalized = re.sub(r'[\u2013\u2014\u2015]', '-', normalized)

    # Collapse multiple whitespace characters to a single space
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized.strip().lower()


def verify_quote_in_text(
    quote: str,
    source_text: str,
    min_length: int = 10,
    fuzzy_threshold: float = 0.95,
) -> Tuple[bool, float]:
    """Deterministically verify whether an extracted quote exists in the source text.

    Args:
        quote: The candidate quote extracted by the LLM.
        source_text: The full captured text of the source.
        min_length: Minimum characters required to consider a quote non-trivial.
        fuzzy_threshold: Minimum token-overlap ratio for minor punctuation variations.

    Returns:
        (is_valid, match_score) where is_valid is True if the quote is substantiated.
    """
    norm_quote = normalize_text(quote)
    norm_source = normalize_text(source_text)

    if not norm_quote or not norm_source:
        return False, 0.0

    if len(norm_quote) < min_length:
        return False, 0.0

    # 1. Exact normalized substring match (ideal case)
    if norm_quote in norm_source:
        return True, 1.0

    # 2. Punctuation-stripped substring match
    clean_quote = re.sub(r'[^\w\s]', '', norm_quote)
    clean_source = re.sub(r'[^\w\s]', '', norm_source)

    if clean_quote and clean_quote in clean_source:
        return True, 0.98

    # 3. Token-level sequence sliding window (guards against minor OCR/linebreak splits)
    quote_tokens = clean_quote.split()
    if not quote_tokens:
        return False, 0.0

    quote_len = len(quote_tokens)
    source_tokens = clean_source.split()

    if quote_len > len(source_tokens):
        return False, 0.0

    best_match = 0.0
    quote_set = set(quote_tokens)

    # Windowed check across source tokens
    for i in range(len(source_tokens) - quote_len + 1):
        window = source_tokens[i : i + quote_len]
        matching = sum(1 for a, b in zip(quote_tokens, window) if a == b)
        ratio = matching / quote_len
        if ratio > best_match:
            best_match = ratio
        if best_match >= fuzzy_threshold:
            return True, best_match

    return False, best_match
