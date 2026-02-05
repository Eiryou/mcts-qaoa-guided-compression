#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quantum-inspired helpers (local, classical simulation)

This module intentionally does NOT claim "quantum speedup".
It provides *quantum-inspired* heuristics that can improve exploration:

- Energy / Hamiltonian view: E = α·Size + β·Distortion + γ·Time + penalties
- "Tunneling" acceptance: occasionally accept worse candidates to escape local minima
- Phase-like interference: keep per-action-path amplitudes to bias sampling
"""

from __future__ import annotations
from dataclasses import dataclass
import cmath
import math
import random
from typing import Dict, Hashable, List, Tuple

@dataclass
class EnergyWeights:
    alpha_size: float = 1.0
    beta_dist: float = 1.0
    gamma_time: float = 0.2
    penalty: float = 5.0

def energy_hamiltonian(size_cost: float, dist_cost: float, time_cost: float,
                      penalty_cost: float = 0.0,
                      w: EnergyWeights = EnergyWeights()) -> float:
    """Energy to minimize (lower is better)."""
    return (w.alpha_size * size_cost) + (w.beta_dist * dist_cost) + (w.gamma_time * time_cost) + (w.penalty * penalty_cost)

def quantum_tunnel_accept(delta_e: float, temp: float, tunnel: float = 0.10) -> bool:
    """Return True if we accept a worse move (delta_e > 0) using annealing + tunneling.
    tunnel adds a small floor probability to jump even when temp is low.
    """
    if delta_e <= 0:
        return True
    if temp <= 1e-9:
        return (random.random() < tunnel)
    p = math.exp(-delta_e / temp)
    return (random.random() < max(p, tunnel))

class PhaseInterferenceSampler:
    """Quantum-inspired amplitude sampler over discrete choices."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.amp: Dict[Hashable, complex] = {}

    def update(self, key: Hashable, energy: float):
        # better (lower) energy => stronger amplitude magnitude
        mag = math.exp(-max(0.0, energy))
        # random phase to simulate interference (can be replaced with deterministic phase)
        phase = self.rng.uniform(0.0, 2.0 * math.pi)
        a = mag * cmath.exp(1j * phase)
        self.amp[key] = self.amp.get(key, 0j) + a

    def sample(self, keys: List[Hashable]) -> Hashable:
        # Probability proportional to |amp|^2; fallback uniform
        weights = []
        for k in keys:
            a = self.amp.get(k, 0j)
            weights.append((a.real * a.real + a.imag * a.imag) + 1e-12)
        s = sum(weights)
        r = self.rng.random() * s
        acc = 0.0
        for k, w in zip(keys, weights):
            acc += w
            if acc >= r:
                return k
        return keys[-1]
