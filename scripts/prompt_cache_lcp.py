"""Token-level longest-common-prefix (LCP) planning for cross-turn KV reuse.

Why this exists (2026-07-25): mlx-server's original cross-turn prompt cache
keys on hash(entire system prompt). The h-uman daemon rebuilds its system
prompt every turn — a stable ~16KB persona head followed by per-turn memory /
exemplar / conversation sections — so the whole-prompt hash never matches and
the cache fully resets on every request: 0% hit rate in production despite
prompt_cache=true. Reusing the longest common *token prefix* instead (the way
stock mlx_lm.server does) lets the stable head hit even though the tail varies.

This module is pure planning — no mlx imports, fully unit-testable
(tests/test_prompt_cache_lcp.py). The server owns all cache mutation.

Modes (mirrors h-uman's off/shadow/live gate discipline):
  off    — legacy hash-based behavior only
  shadow — legacy behavior, plus log what LCP *would* have reused
  live   — trim the cache to the LCP and prefill only the suffix
"""

MODES = ("off", "shadow", "live")


def parse_mode(raw, default="shadow"):
    """Parse an LCP mode string. Unknown values fail closed to 'off'."""
    if raw is None:
        return default
    val = str(raw).strip().lower()
    if val in MODES:
        return val
    if val in ("", "default"):
        return default
    return "off"


def common_prefix_len(a, b):
    """Length of the longest common prefix of two token sequences."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def plan_reuse(prev_tokens, new_tokens, cache_size):
    """Plan cache reuse for a new request.

    prev_tokens: token ids whose KV state the cache holds as its prefix
                 (the previous request's full prompt; the cache may hold
                 additional generated tokens beyond them — cache_size covers
                 those).
    new_tokens:  the new request's full prompt token ids.
    cache_size:  current cache offset in tokens.

    Returns (reuse, trim_amount):
      reuse       — tokens of KV state to keep (callers prefill new_tokens[reuse:])
      trim_amount — tokens to trim from the cache tail before prefill

    Invariants:
      - at least one suffix token is always left to prefill (mlx_lm requires
        a non-empty prompt), so reuse <= len(new_tokens) - 1
      - reuse == 0 means full reset: trim the whole cache
      - a cache smaller than the previous prompt (unexpected state) plans a
        full reset rather than guessing
    """
    if not prev_tokens or not new_tokens or cache_size <= 0:
        return 0, max(0, cache_size)
    if cache_size < len(prev_tokens):
        # Cache no longer holds the full previous prompt — state is
        # inconsistent with our bookkeeping; reset rather than risk reusing
        # KV entries that don't correspond to prev_tokens.
        return 0, cache_size

    reuse = common_prefix_len(prev_tokens, new_tokens)
    reuse = min(reuse, len(new_tokens) - 1)
    if reuse <= 0:
        return 0, cache_size
    return reuse, cache_size - reuse
