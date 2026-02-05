#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apps/two_phase.py

Two-phase search pattern:
  - Probe: cheap/fast evaluation to pick a strong candidate
  - Final: apply best candidate once on full-resolution/full-length input
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Any, Dict, Optional, Tuple, List


@dataclass(frozen=True)
class TwoPhaseConfig:
    max_probe_candidates: int = 30
    early_stop_score: Optional[float] = None


def run_two_phase_search(
    cfg: TwoPhaseConfig,
    probe_generate_candidates: Callable[[], Iterable[Any]],
    probe_evaluate: Callable[[Any], Tuple[float, Dict[str, Any]]],
    final_apply: Callable[[Any], Any],
) -> Tuple[Optional[Any], List[Dict[str, Any]]]:
    """Returns (final_result, probe_log)."""
    best = None
    best_score = float("-inf")
    probe_log: List[Dict[str, Any]] = []

    for i, cand in enumerate(probe_generate_candidates()):
        if i >= cfg.max_probe_candidates:
            break

        score, info = probe_evaluate(cand)
        row = {"i": i, "score": float(score)}
        if info:
            row.update(info)
        probe_log.append(row)

        if score > best_score:
            best_score = score
            best = cand

        if cfg.early_stop_score is not None and best_score >= cfg.early_stop_score:
            break

    if best is None:
        return None, probe_log

    final_result = final_apply(best)
    return final_result, probe_log
