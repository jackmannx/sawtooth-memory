"""Shared Dual-Target Externalization helpers for async and sync managers."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Hashable

from .config import ContextManagerConfig
from .fold_unit import fold_lines, remove_fold_lines
from .intent_planner import PromptIntentPlan
from .novelty import residualize
from .observation_crush import crush_observation
from .state import ArchivalMemory, DTEState, EntityLedger, MemoryState, Message


@dataclass(frozen=True)
class ConsolidationPrep:
    """Prepared residual consolidation payload, or a novelty-skip outcome."""

    message: Message | None = None
    novelty_skipped: bool = False


def prompt_turn_key(state: MemoryState) -> tuple[Hashable, ...]:
    """Identity for one spend-accounting turn (L1 + fold boundary)."""
    messages = state.l1_working.messages
    last_id = messages[-1].id if messages else None
    return (len(messages), last_id, state.dte.fold_cycles, state.dte.consolidation_cycles)


def account_main_prompt_tokens(
    dte: DTEState,
    prompt_tokens: int,
    *,
    previous_key: tuple[Hashable, ...] | None,
    current_key: tuple[Hashable, ...],
) -> tuple[Hashable, ...]:
    """Accumulate prompt spend once per turn key (not every build_prompt call)."""
    if previous_key != current_key:
        dte.main_prompt_tokens += prompt_tokens
    return current_key


def apply_observation_crush(
    role: str,
    content: str,
    *,
    config: ContextManagerConfig,
    count_text: Callable[[str], int],
    cache: OrderedDict[str, str],
    dte: DTEState,
) -> str:
    """Crush tool observations and retain reversible originals in *cache*."""
    if not config.enable_observation_crush or role != "tool":
        return content

    crushed = crush_observation(
        content,
        count_text=count_text,
        min_tokens=config.obs_crush_min_tokens,
    )
    if not crushed.crushed or not crushed.cache_id:
        return crushed.content

    cache[crushed.cache_id] = content
    cache.move_to_end(crushed.cache_id)
    while len(cache) > config.obs_cache_max_entries:
        cache.popitem(last=False)
    dte.observation_tokens_saved += crushed.tokens_saved
    return crushed.content


def pool_content_fingerprint(
    entities: EntityLedger, archive: ArchivalMemory
) -> str:
    """Stable fingerprint used to skip redundant pool merges."""
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(len(entities.entities)).encode())
    digest.update(archive.narrative.encode())
    for key in sorted(entities.entities):
        digest.update(key.encode())
        history = entities.entities[key]
        if history:
            digest.update(history[-1].encode())
            digest.update(str(len(history)).encode())
    return digest.hexdigest()


def merge_pool_into_state(
    state: MemoryState,
    pool_entities: EntityLedger,
    pool_archive: ArchivalMemory,
) -> None:
    """Merge shared pool entities/narratives into local session state."""
    for key, history in pool_entities.entities.items():
        for value in history:
            state.l1_5_entities.upsert({key: value})

    pool_narrative = pool_archive.narrative.strip()
    if not pool_narrative:
        return

    local_narrative = state.l2_archival.narrative.strip()
    if not local_narrative:
        state.l2_archival.narrative = pool_narrative
    elif pool_narrative not in local_narrative:
        state.l2_archival.append_narrative(pool_narrative)


def apply_fold_delta_to_pool(
    *,
    session_id: str,
    fold_stub: str,
    entity_keys: tuple[str, ...],
    local_entities: EntityLedger,
    shared_entities: EntityLedger,
    shared_archive: ArchivalMemory,
) -> None:
    """Apply a fold stub and latest entity values onto shared pool state."""
    for key in entity_keys:
        value = local_entities.get_latest(key)
        if value is not None:
            shared_entities.upsert({key: value})
    shared_archive.append_narrative(f"[origin:{session_id}] {fold_stub}")


def prepare_consolidation(
    state: MemoryState,
    config: ContextManagerConfig,
    intent_plan: PromptIntentPlan,
    *,
    count_text: Callable[[str], int],
    count_message: Callable[[Message], int],
    require_sync_flag: bool = False,
) -> ConsolidationPrep | None:
    """Return a consolidation payload when debt, novelty, and spend allow it.

    Returns ``None`` when consolidation should not run. A ``ConsolidationPrep``
    with ``novelty_skipped=True`` means folds were cleared without an LLM call.
    """
    if config.compression_mode != "dte" or not config.consolidation_on_idle:
        return None
    if require_sync_flag and not config.enable_sync_consolidation:
        return None
    if state.dte.consolidation_queued:
        return None

    debt = state.dte.narrative_debt_tokens
    if debt <= 0:
        return None
    if debt < config.narrative_debt_trigger_tokens and not intent_plan.prefers_narrative:
        return None

    folds = fold_lines(state.l2_archival.narrative)
    if not folds:
        state.dte.narrative_debt_tokens = 0
        return None

    source = "\n".join(folds)
    residual = source
    if config.enable_novelty_filter:
        novelty = residualize(
            source,
            state.l1_5_entities,
            state.l2_archival.narrative,
            count_text=count_text,
        )
        residual = novelty.residual
        if not residual or novelty.residual_ratio < config.novelty_min_residual:
            state.l2_archival.narrative = remove_fold_lines(state.l2_archival.narrative)
            state.dte.narrative_debt_tokens = 0
            state.dte.folds_since_narrative = 0
            state.dte.novelty_skips += 1
            return ConsolidationPrep(novelty_skipped=True)

    guideline = config.compression_guideline or (
        "Consolidate these structured fold outcomes into one dense, "
        "causal narrative. Do not repeat exact identifiers already protected "
        "by the entity ledger."
    )
    content = f"{guideline}\n\n{residual}"
    message = Message(role="system", content=content)
    message.token_count = count_message(message)

    allowance = int(state.dte.main_prompt_tokens * config.background_spend_ratio)
    projected = state.dte.background_llm_input_tokens + message.token_count
    if projected > allowance:
        return None

    return ConsolidationPrep(message=message)
