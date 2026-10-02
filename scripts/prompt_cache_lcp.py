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

# Below this LCP, reusing a slot isn't worth evicting whatever it holds —
# interleaved callers on :8741 share only a ~6-9 token chat-template
# preamble, and evicting the daemon's ~16KB persona head to save that
# preamble is exactly the failure the slot pool exists to prevent.
DEFAULT_SLOT_FLOOR = 64

# Each slot pins real KV memory once prefilled, so the pool stays bounded no
# matter what the config asks for. Sizing (2026-10-02, prod GLM-4.5-Air-4bit,
# kv_bits=8, group 64): 46 layers x 2 (K,V) x 8 kv heads x 128 head_dim =
# 94,208 elements/token; 8-bit packed + bf16 scale/bias per 64 = 1.0625 B/elem
# => ~100 KB/token. A ~5k-token reply prefix (+ generation, 256-token step
# rounding) pins ~0.5-0.6 GB, so 6 slots ~= 3.6 GB beside ~57 GB of weights.
# Note a slot keeps the PEAK allocation it ever reached (trim only moves the
# offset), so a slot that once held an 8k prompt pins ~0.8 GB.
#
# Why 6 (was 3 in prod): a measured 28% of main-reply calls were evicted with
# 3 slots — the slots churned under small judge/extract/proposer prompts. The
# scratch slot (below) takes the small prompts off the pool entirely; 6
# regular slots then cover the ~6 active contacts' reply prefixes.
MAX_SLOTS = 8
DEFAULT_SLOTS = 6

# Prompts shorter than this many tokens are routed to a dedicated SCRATCH slot
# and never touch (reuse-trim OR evict) a regular slot. Reusing a regular slot
# for a short prompt trims it to the LCP, which destroys a long reply prefix
# just as surely as evicting it. 0 disables the scratch slot (legacy routing).
DEFAULT_SCRATCH_MAX_TOKENS = 400
MAX_SCRATCH_MAX_TOKENS = 8192


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


def parse_slots(raw, default=DEFAULT_SLOTS):
    """Parse the regular slot-pool size, clamped to [1, MAX_SLOTS]."""
    try:
        n = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return max(1, min(n, MAX_SLOTS))


def parse_scratch_max_tokens(raw, default=DEFAULT_SCRATCH_MAX_TOKENS):
    """Parse the scratch-slot threshold, clamped to [0, MAX_SCRATCH_MAX_TOKENS].

    0 (or any negative) disables the scratch slot. Junk falls back to default.
    """
    if raw is None:
        return default
    try:
        n = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return max(0, min(n, MAX_SCRATCH_MAX_TOKENS))


EVICT_POLICIES = ("value", "lru")


def parse_evict_policy(raw, default="value"):
    """Parse the eviction policy. Unknown values fall back to the default.

    'lru' restores the pre-2026-07-27 behavior and exists as a one-env-var
    rollback, not as a recommended setting — it is the policy the eviction
    logging showed to be broken.
    """
    if raw is None:
        return default
    val = str(raw).strip().lower()
    return val if val in EVICT_POLICIES else default


def slot_evict_value(prev_tokens, age):
    """How much we lose by discarding this slot. Higher = protect it harder.

    This is GreedyDual-Size: value = cached tokens / (age + 1), where `age` is
    how many requests ago the slot was last used. The numerator is the real
    cost of eviction — the tokens we would have to re-prefill if this slot were
    reused — and the denominator discounts by staleness as a proxy for how
    likely that reuse is.

    Why plain LRU was wrong (measured in prod 2026-07-27): eviction ran on
    min(last_used), which is entirely size-indifferent, so a 10-token probe
    evicted a 4143-token cached prefix. Every observed eviction was at `idle 2
    reqs` with 3 slots — the pool cycled completely every 3 requests and
    nothing survived long enough to be reused. Hit rate stayed high (74.8%)
    because the surviving hits were trivia, while token-weighted reuse
    collapsed to 34.4% against 59.6% in shadow.

    The decay is self-limiting, so no entry is immortal and no arbitrary
    staleness cutoff is needed: a 4227-token slot outranks a fresh 200-token
    one only while 4227/(age+1) > 200, i.e. for about 20 idle requests. A
    genuinely dead conversation ages out on its own.

    Returns 0.0 for an empty slot so empty slots are always the first victims.
    """
    size = len(prev_tokens) if prev_tokens else 0
    if size <= 0:
        return 0.0
    return size / (float(age) + 1.0)


def choose_slot(slot_prev_tokens, slot_last_used, new_tokens,
                floor=DEFAULT_SLOT_FLOOR, policy="value"):
    """Pick which cache slot serves a new request.

    slot_prev_tokens: per-slot token lists (None/[] = empty slot)
    slot_last_used:   per-slot monotonic use counters (higher = more recent)
    new_tokens:       the new request's full prompt token ids
    floor:            minimum LCP worth reusing a slot for
    policy:           'value' (default) evicts the least valuable slot per
                      slot_evict_value; 'lru' is the legacy rollback.

    Returns (slot_idx, reuse):
      reuse > 0  — reuse `reuse` prefix tokens of slot slot_idx
      reuse == 0 — reset slot slot_idx and prefill from scratch

    Eviction policy: if no slot clears the floor, recycle an EMPTY slot
    first, then the LEAST VALUABLE one (fewest cached tokens, discounted by
    staleness) — never the best-LCP slot. This is what lets a big stable-head
    slot survive interleaved small-probe traffic that only matches the
    chat-template preamble.

    That last sentence was the stated intent from the start, but until
    2026-07-27 the victim was chosen by min(last_used) — size-indifferent —
    and prod logging showed it failing exactly as described: a 10-token probe
    evicted a 4143-token prefix. Selecting the slot to REUSE is unchanged;
    only the choice of victim moved.
    """
    n = len(slot_prev_tokens)
    if n == 0:
        return 0, 0
    cap = max(0, len(new_tokens) - 1)

    best_idx, best_lcp = 0, -1
    for i, prev in enumerate(slot_prev_tokens):
        lcp = min(common_prefix_len(prev, new_tokens), cap) if prev else 0
        # Tie-break to the most recently used so the LRU slot stays
        # available for eviction.
        if lcp > best_lcp or (lcp == best_lcp and slot_last_used[i] > slot_last_used[best_idx]):
            best_idx, best_lcp = i, lcp
    if best_lcp >= floor and best_lcp > 0:
        return best_idx, best_lcp

    for i, prev in enumerate(slot_prev_tokens):
        if not prev:
            return i, 0

    if policy == "lru":
        return min(range(n), key=lambda i: slot_last_used[i]), 0

    # `now` is the newest counter seen: the current request has not been
    # counted yet at selection time, so this is the best available clock.
    now = max(slot_last_used)
    victim = min(
        range(n),
        key=lambda i: (slot_evict_value(slot_prev_tokens[i], now - slot_last_used[i]),
                       slot_last_used[i]),  # tie-break: oldest first, deterministic
    )
    return victim, 0


def route_request(slot_prev_tokens, slot_last_used, new_tokens,
                  scratch_idx=None, scratch_max_tokens=0,
                  floor=DEFAULT_SLOT_FLOOR, policy="value"):
    """Pick a slot for a request, honouring the scratch slot.

    scratch_idx:        index of the scratch slot in the pool, or None (no
                        scratch slot — identical to choose_slot over the pool).
    scratch_max_tokens: prompts with len(new_tokens) < this go to the scratch
                        slot; 0 disables scratch routing.

    Returns (slot_idx, reuse, is_scratch).

    Small prompts ALWAYS land on the scratch slot (reusing its prefix when
    that clears the floor, else resetting it). They never reuse a regular
    slot either: plan_reuse would trim that slot to the short LCP, destroying
    its long prefix. Long prompts pick among the regular slots only, with the
    usual best-LCP / least-valuable-victim logic, and never see the scratch.
    """
    n = len(slot_prev_tokens)
    scratch_on = (scratch_idx is not None and scratch_max_tokens > 0
                  and 0 <= scratch_idx < n and n >= 2)
    if scratch_on and len(new_tokens) < scratch_max_tokens:
        prev = slot_prev_tokens[scratch_idx]
        cap = max(0, len(new_tokens) - 1)
        lcp = min(common_prefix_len(prev, new_tokens), cap) if prev else 0
        return scratch_idx, (lcp if lcp >= floor and lcp > 0 else 0), True
    if not scratch_on:
        idx, reuse = choose_slot(slot_prev_tokens, slot_last_used, new_tokens,
                                 floor=floor, policy=policy)
        return idx, reuse, False
    regular = [i for i in range(n) if i != scratch_idx]
    sub_idx, reuse = choose_slot([slot_prev_tokens[i] for i in regular],
                                 [slot_last_used[i] for i in regular],
                                 new_tokens, floor=floor, policy=policy)
    return regular[sub_idx], reuse, False
