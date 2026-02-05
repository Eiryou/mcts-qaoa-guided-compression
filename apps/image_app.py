#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
apps/image_app.py

MCTS-ish Image Optimizer (lightweight)
- Probe: downscaled SSIM scoring for many candidates
- Final: encode once at full resolution
- Correct output extension + MIME type

Fixes / Improvements:
- Streamlit deprecation: use_container_width -> width="stretch"
- PhaseInterferenceSampler API alignment: update(key, energy) then sample(keys)
- Robustness: skip candidates that fail to encode/decode (e.g., AVIF plugin missing)
- Keep running even if some candidates are unsupported

NEW (Quantum Computing Fusion, Local-only):
- "Quantum circuit" simulated QAOA (p=1) using NumPy statevector simulation
- QAOA acts as a candidate generator over discrete (fmt, quality) choices
- It uses real probe measurements (size, SSIM) as energies, then biases candidate choices
  via a QAOA circuit (phase separator + mixer) optimized on a small grid.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import streamlit as st
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from apps.quantum_inspired import PhaseInterferenceSampler
from apps.common_guard import GuardConfig, run_guard_or_stop, ConcurrencyGuard, show_exception
from apps.utils import guess_mime, safe_suffix_for_image


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class Candidate:
    fmt: str
    quality: int
    max_side: int


# =============================================================================
# Basic image helpers
# =============================================================================

def _to_gray_float(im: Image.Image) -> np.ndarray:
    g = im.convert("L")
    arr = np.asarray(g).astype(np.float32) / 255.0
    return arr


def _resize_max_side(im: Image.Image, max_side: int) -> Image.Image:
    w, h = im.size
    m = max(w, h)
    if m <= max_side:
        return im
    scale = max_side / float(m)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return im.resize((nw, nh), resample=Image.LANCZOS)


def _encode_image(im: Image.Image, fmt: str, quality: int) -> bytes:
    out = io.BytesIO()
    fmt_u = fmt.upper()
    save_kwargs = {}

    if fmt.lower() in ("jpeg", "jpg"):
        save_kwargs.update(dict(format="JPEG", quality=int(quality), optimize=True, progressive=True))
    elif fmt.lower() == "webp":
        save_kwargs.update(dict(format="WEBP", quality=int(quality), method=6))
    elif fmt.lower() == "avif":
        # Pillow-avif-plugin required on some envs; if absent, this may fail.
        save_kwargs.update(dict(format="AVIF", quality=int(quality)))
    elif fmt.lower() == "png":
        # PNG is lossless; quality ignored
        save_kwargs.update(dict(format="PNG", optimize=True, compress_level=9))
    else:
        save_kwargs.update(dict(format=fmt_u))

    im.save(out, **save_kwargs)
    return out.getvalue()


def _score(orig_small: Image.Image, cand_small: Image.Image, out_bytes: bytes, time_penalty: float = 0.0) -> float:
    a = _to_gray_float(orig_small)
    b = _to_gray_float(cand_small)

    # Ensure same size
    if a.shape != b.shape:
        b = np.asarray(
            cand_small.convert("L").resize(orig_small.size, Image.LANCZOS)
        ).astype(np.float32) / 255.0

    s = float(ssim(a, b, data_range=1.0))
    size_kb = len(out_bytes) / 1024.0

    # We want small size, high SSIM. Score higher is better.
    return (s * 1000.0) - (size_kb * 1.0) - (time_penalty * 50.0)


# =============================================================================
# Quantum-inspired (existing) helper for format energy
# =============================================================================

def _fmt_energy(fmt: str) -> float:
    """Heuristic 'energy' for format preference (lower is better)."""
    fl = fmt.lower()
    if fl in ("avif", "webp"):
        return 0.0
    if fl in ("jpg", "jpeg"):
        return 0.6
    return 1.0


# =============================================================================
# Quantum circuit simulated QAOA (p=1) over discrete choices
# =============================================================================

def _rx_gate(beta: float) -> np.ndarray:
    """
    Rx(2*beta) gate matrix.
    """
    c = np.cos(beta)
    s = -1j * np.sin(beta)
    return np.array([[c, s],
                     [s, c]], dtype=np.complex128)


def _apply_1q_gate_statevec(state: np.ndarray, gate: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """
    Apply a 1-qubit gate to a statevector of size 2^n.
    qubit: 0 is LSB (rightmost bit) for indexing consistency.
    """
    dim = 1 << n_qubits
    state = state.reshape([2] * n_qubits)

    # Move target axis to front for easy multiplication
    axes = list(range(n_qubits))
    axes[0], axes[qubit] = axes[qubit], axes[0]
    state_t = np.transpose(state, axes).reshape(2, -1)

    state_t = gate @ state_t

    # Restore axes
    state_t = state_t.reshape([2] + [2] * (n_qubits - 1))
    inv_axes = np.argsort(axes)
    out = np.transpose(state_t, inv_axes).reshape(dim)
    return out


def _qaoa_p1_distribution(energies: np.ndarray,
                          gamma_grid: np.ndarray,
                          beta_grid: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    """
    Simulate QAOA p=1 on a diagonal cost Hamiltonian with energies[z].
    Start from |+>^n, apply phase e^{-i gamma E(z)}, then Rx(2*beta) on all qubits.
    Choose gamma, beta by minimizing expected energy.
    Return: (probs, best_gamma, best_beta, best_expE)
    """
    energies = np.asarray(energies, dtype=np.float64)
    dim = energies.shape[0]
    n_qubits = int(np.log2(dim))

    # |+>^n statevector (uniform)
    base = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)

    best_expE = float("inf")
    best_gamma = float(gamma_grid[0])
    best_beta = float(beta_grid[0])
    best_probs = np.ones(dim, dtype=np.float64) / dim

    for gamma in gamma_grid:
        # phase separator (diagonal)
        phase = np.exp(-1j * gamma * energies).astype(np.complex128)

        for beta in beta_grid:
            stv = base * phase

            # mixer: Rx on each qubit
            rx = _rx_gate(beta)
            for qb in range(n_qubits):
                stv = _apply_1q_gate_statevec(stv, rx, qb, n_qubits)

            probs = (stv.real * stv.real + stv.imag * stv.imag).astype(np.float64)
            probs = probs / max(probs.sum(), 1e-18)

            expE = float((probs * energies).sum())
            if expE < best_expE:
                best_expE = expE
                best_gamma = float(gamma)
                best_beta = float(beta)
                best_probs = probs

    return best_probs, best_gamma, best_beta, best_expE


def _build_discrete_quality_levels(q_min: int, q_max: int, levels: int) -> List[int]:
    if levels <= 1:
        return [int(max(1, min(100, q_max)))]
    qs = np.linspace(q_min, q_max, num=int(levels))
    qs = [int(round(x)) for x in qs]
    # clamp
    qs = [max(1, min(100, q)) for q in qs]
    # unique + keep order
    out = []
    for q in qs:
        if q not in out:
            out.append(q)
    return out


def _quantum_generate_candidates_qaoa(
    orig_im: Image.Image,
    orig_small: Image.Image,
    probe_max_side: int,
    allowed_formats: List[str],
    q_levels: List[int],
    min_ssim: float,
    alpha_size: float = 1.0,
    beta_dist: float = 600.0,
    # QAOA grid (small, local-only)
    gamma_steps: int = 11,
    beta_steps: int = 11,
    top_k: int = 8,
) -> Tuple[List[Candidate], Dict[str, float]]:
    """
    Quantum computing fusion:
    - Enumerate discrete combos (fmt, q) -> energies via REAL probe encode+SSIM
    - Run QAOA(p=1) statevector simulation to produce a probability distribution
    - Take top-K bitstrings as high-priority candidates

    energies = alpha*size_kb + beta*(1-ssim) + penalty(for SSIM < floor) + penalty(for unsupported encode)

    Returns candidates and debug stats.
    """
    # Map combos to indices
    combos: List[Tuple[str, int]] = []
    for f in allowed_formats:
        fl = f.lower()
        # PNG ignores quality; still allow once
        if fl == "png":
            combos.append((f, 100))
        else:
            for q in q_levels:
                combos.append((f, int(q)))

    # Limit dimension: keep it small (<= 64 recommended for speed)
    # If too many combos, randomly subsample but keep at least each format represented.
    max_dim = 64
    if len(combos) > max_dim:
        # Keep 1 PNG + random subset of others
        rng = random.Random(1337)
        # ensure each format appears
        by_fmt: Dict[str, List[Tuple[str, int]]] = {}
        for c in combos:
            by_fmt.setdefault(c[0], []).append(c)
        picked = []
        for f, lst in by_fmt.items():
            picked.append(rng.choice(lst))
        # fill
        remain = [c for c in combos if c not in picked]
        rng.shuffle(remain)
        while len(picked) < max_dim and remain:
            picked.append(remain.pop())
        combos = picked

    # Make dimension power of 2 for QAOA simulation
    n = len(combos)
    n_qubits = int(np.ceil(np.log2(max(1, n))))
    dim = 1 << n_qubits

    # Compute energies for each combo (real probe). Unsupported or fail => huge energy.
    energies = np.full(dim, 1e6, dtype=np.float64)

    skipped_encode = 0
    skipped_decode = 0
    measured = 0

    for idx, (fmt, q) in enumerate(combos):
        try:
            test_im = _resize_max_side(orig_im, probe_max_side)
            out_b = _encode_image(test_im, fmt, q)
        except Exception:
            skipped_encode += 1
            continue

        try:
            cand_im = Image.open(io.BytesIO(out_b)).convert("RGB")
        except Exception:
            skipped_decode += 1
            continue

        cand_small = _resize_max_side(cand_im, probe_max_side)

        try:
            s_val = float(ssim(_to_gray_float(orig_small), _to_gray_float(cand_small), data_range=1.0))
        except Exception:
            # treat as bad
            s_val = 0.0

        size_kb = len(out_b) / 1024.0
        dist = (1.0 - s_val)

        # penalty if below SSIM floor
        penalty = 0.0
        if s_val < float(min_ssim):
            penalty = 50.0 + (float(min_ssim) - s_val) * 200.0

        E = (alpha_size * size_kb) + (beta_dist * dist) + penalty
        energies[idx] = float(E)
        measured += 1

    # QAOA grid search
    gamma_grid = np.linspace(0.0, 2.5, num=max(3, int(gamma_steps))).astype(np.float64)
    beta_grid = np.linspace(0.0, np.pi / 2.0, num=max(3, int(beta_steps))).astype(np.float64)

    probs, best_gamma, best_beta, best_expE = _qaoa_p1_distribution(energies, gamma_grid, beta_grid)

    # pick top-k indices by probability, but only from actual combos range [0,n)
    order = np.argsort(-probs)  # desc
    out_cands: List[Candidate] = []
    used = set()

    for j in order:
        j = int(j)
        if j >= n:
            continue
        fmt, q = combos[j]
        key = (fmt.lower(), int(q))
        if key in used:
            continue
        used.add(key)
        out_cands.append(Candidate(fmt=fmt, quality=int(q), max_side=probe_max_side))
        if len(out_cands) >= int(max(1, top_k)):
            break

    stats = dict(
        qaoa_dim=float(dim),
        qaoa_ncombos=float(n),
        qaoa_measured=float(measured),
        qaoa_skipped_encode=float(skipped_encode),
        qaoa_skipped_decode=float(skipped_decode),
        qaoa_best_gamma=float(best_gamma),
        qaoa_best_beta=float(best_beta),
        qaoa_best_expE=float(best_expE),
    )
    return out_cands, stats


# =============================================================================
# Candidate generation (classical + quantum-inspired + quantum-circuit)
# =============================================================================

def _make_candidates(
    fmt_choices: List[str],
    q_min: int,
    q_max: int,
    n: int,
    probe_max_side: int,
    *,
    use_qi: bool = False,
    sampler: Optional[PhaseInterferenceSampler] = None,
) -> List[Candidate]:
    """
    Classical candidate generator.
    (Quantum-circuit candidates are generated separately and merged.)
    """
    cands: List[Candidate] = []

    def pick_fmt() -> str:
        if use_qi and sampler is not None:
            for f in fmt_choices:
                sampler.update(f, _fmt_energy(f))
            return str(sampler.sample(fmt_choices))
        return random.choice(fmt_choices)

    for _ in range(int(n)):
        fmt = pick_fmt()
        if fmt.lower() == "png":
            q = 100
        else:
            q = random.randint(int(q_min), int(q_max))
        cands.append(Candidate(fmt=fmt, quality=int(q), max_side=int(probe_max_side)))

    return cands


# =============================================================================
# Streamlit App
# =============================================================================

def render_image_app(cfg: GuardConfig) -> None:
    st.header("Image Optimizer (Probe → Final)")
    st.caption("Search-based: tries many candidates on a downscaled probe, then encodes once at full resolution.")

    up = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp", "avif"])
    if not up:
        return

    run_guard_or_stop(cfg, up)

    col1, col2, col3 = st.columns(3)
    with col1:
        keep_ext = st.checkbox("Keep original extension (default)", value=True)
    with col2:
        probe_max_side = st.slider("Probe max_side", 128, 512, 256, step=32)
    with col3:
        final_max_side = st.slider("Final max_side", 256, 2048, 1024, step=64)

    # Output format choices
    orig_suffix = "." + up.name.split(".")[-1].lower()
    orig_fmt = "jpeg" if orig_suffix in (".jpg", ".jpeg") else orig_suffix.lstrip(".")
    fmt_choices = ["jpeg", "webp", "avif", "png"]

    if keep_ext and orig_fmt in fmt_choices:
        allowed = [orig_fmt]
    else:
        allowed = st.multiselect("Candidate formats", fmt_choices, default=["jpeg", "webp", "avif"])

    q_min, q_max = st.slider("Quality range (lossy)", 1, 100, (35, 92))
    n_trials = st.slider("Probe trials (classical)", 10, 400, 80, step=10)

    # quantum-inspired sampling (existing)
    use_qi = st.checkbox("Quantum-inspired sampling (phase interference)", value=True)
    seed = st.number_input("Quantum-inspired seed", min_value=0, max_value=2**31 - 1, value=1337, step=1)
    _qi_sampler = PhaseInterferenceSampler(seed=int(seed))

    min_ssim = st.slider("Min SSIM (probe)", 0.60, 0.99, 0.93, step=0.01)

    # --- NEW: Quantum computing (QAOA circuit simulation) controls
    st.subheader("Quantum Computing Fusion (QAOA circuit simulation)")
    use_qaoa = st.checkbox("Enable QAOA(p=1) quantum circuit candidate generator", value=True)
    qaoa_topk = st.slider("QAOA top-K candidates", 2, 24, 10, step=1)
    qaoa_qlevels = st.slider("Discrete quality levels for QAOA", 3, 16, 8, step=1)
    qaoa_gamma_steps = st.slider("QAOA gamma grid steps", 5, 31, 11, step=2)
    qaoa_beta_steps = st.slider("QAOA beta grid steps", 5, 31, 11, step=2)
    # weights for energy
    colw1, colw2 = st.columns(2)
    with colw1:
        alpha_size = st.slider("Energy weight: size (alpha)", 0.1, 5.0, 1.0, step=0.1)
    with colw2:
        beta_dist = st.slider("Energy weight: distortion (beta)", 50.0, 2000.0, 600.0, step=50.0)

    if st.button("Optimize (Probe → Final)"):
        try:
            with ConcurrencyGuard(cfg):
                # Load original
                orig_bytes = up.getvalue()
                orig_im = Image.open(io.BytesIO(orig_bytes)).convert("RGB")

                st.image(orig_im, caption=f"Original ({len(orig_bytes)/1024:.1f} KB)", width="stretch")

                # Probe downscale baseline
                orig_small = _resize_max_side(orig_im, probe_max_side)

                # --- Quantum-circuit (QAOA) candidates (generated BEFORE classical probe loop)
                qaoa_cands: List[Candidate] = []
                qaoa_stats: Dict[str, float] = {}
                if use_qaoa and len(allowed) > 0:
                    q_levels = _build_discrete_quality_levels(int(q_min), int(q_max), int(qaoa_qlevels))
                    qaoa_cands, qaoa_stats = _quantum_generate_candidates_qaoa(
                        orig_im=orig_im,
                        orig_small=orig_small,
                        probe_max_side=int(probe_max_side),
                        allowed_formats=list(allowed),
                        q_levels=q_levels,
                        min_ssim=float(min_ssim),
                        alpha_size=float(alpha_size),
                        beta_dist=float(beta_dist),
                        gamma_steps=int(qaoa_gamma_steps),
                        beta_steps=int(qaoa_beta_steps),
                        top_k=int(qaoa_topk),
                    )

                # Classical candidates
                classical_cands = _make_candidates(
                    list(allowed),
                    int(q_min),
                    int(q_max),
                    int(n_trials),
                    int(probe_max_side),
                    use_qi=bool(use_qi),
                    sampler=_qi_sampler,
                )

                # Merge (QAOA first -> priority)
                cands = list(qaoa_cands) + list(classical_cands)

                # Dedup (fmt, quality, max_side)
                uniq: List[Candidate] = []
                seen = set()
                for c in cands:
                    k = (c.fmt.lower(), int(c.quality), int(c.max_side))
                    if k in seen:
                        continue
                    seen.add(k)
                    uniq.append(c)
                cands = uniq

                # Probe loop
                best = None
                best_score = -1e18
                prog = st.progress(0)

                skipped_encode = 0
                skipped_decode = 0
                skipped_ssim = 0

                for i, c in enumerate(cands, 1):
                    test_im = _resize_max_side(orig_im, c.max_side)

                    # Robust encode: skip unsupported formats
                    try:
                        out_b = _encode_image(test_im, c.fmt, c.quality)
                    except Exception:
                        skipped_encode += 1
                        prog.progress(i / len(cands))
                        continue

                    # Robust decode
                    try:
                        cand_im = Image.open(io.BytesIO(out_b)).convert("RGB")
                    except Exception:
                        skipped_decode += 1
                        prog.progress(i / len(cands))
                        continue

                    cand_small = _resize_max_side(cand_im, probe_max_side)

                    # Score
                    sc = _score(orig_small, cand_small, out_b)

                    # hard SSIM floor
                    try:
                        s_val = float(ssim(_to_gray_float(orig_small), _to_gray_float(cand_small), data_range=1.0))
                    except Exception:
                        skipped_ssim += 1
                        prog.progress(i / len(cands))
                        continue

                    if s_val >= min_ssim and sc > best_score:
                        best_score = sc
                        best = (c, s_val, len(out_b))

                    prog.progress(i / len(cands))

                if best is None:
                    st.error(
                        "No candidate met the SSIM floor on probe. "
                        "Try lowering Min SSIM or increasing trials. "
                        f"(skipped: encode={skipped_encode}, decode={skipped_decode}, ssim={skipped_ssim})"
                    )
                    # show QAOA stats if enabled
                    if use_qaoa and qaoa_stats:
                        st.info(
                            f"QAOA stats: combos={int(qaoa_stats.get('qaoa_ncombos',0))} "
                            f"measured={int(qaoa_stats.get('qaoa_measured',0))} "
                            f"skipped_encode={int(qaoa_stats.get('qaoa_skipped_encode',0))} "
                            f"skipped_decode={int(qaoa_stats.get('qaoa_skipped_decode',0))} "
                            f"best_gamma={qaoa_stats.get('qaoa_best_gamma',0):.3f} "
                            f"best_beta={qaoa_stats.get('qaoa_best_beta',0):.3f} "
                            f"best_expE={qaoa_stats.get('qaoa_best_expE',0):.3f}"
                        )
                    return

                c_best, s_best, size_best = best
                st.success(
                    f"Probe best: fmt={c_best.fmt} quality={c_best.quality} | "
                    f"SSIM≈{s_best:.4f} | probe_size={size_best/1024:.1f} KB | "
                    f"skipped: encode={skipped_encode}, decode={skipped_decode}, ssim={skipped_ssim}"
                )

                if use_qaoa and qaoa_stats:
                    st.info(
                        f"QAOA stats: combos={int(qaoa_stats.get('qaoa_ncombos',0))} "
                        f"measured={int(qaoa_stats.get('qaoa_measured',0))} "
                        f"skipped_encode={int(qaoa_stats.get('qaoa_skipped_encode',0))} "
                        f"skipped_decode={int(qaoa_stats.get('qaoa_skipped_decode',0))} "
                        f"best_gamma={qaoa_stats.get('qaoa_best_gamma',0):.3f} "
                        f"best_beta={qaoa_stats.get('qaoa_best_beta',0):.3f} "
                        f"best_expE={qaoa_stats.get('qaoa_best_expE',0):.3f}"
                    )

                # Final encode once (full, possibly downscaled)
                final_im = _resize_max_side(orig_im, final_max_side)
                try:
                    out_final = _encode_image(final_im, c_best.fmt, c_best.quality)
                except Exception:
                    st.error(
                        f"Final encode failed for format={c_best.fmt}. "
                        "Try disabling that format (e.g., AVIF) or install encoder plugins."
                    )
                    raise

                orig_size = len(orig_bytes)
                new_size = len(out_final)
                red = (1.0 - (new_size / float(orig_size))) * 100.0 if orig_size else 0.0
                st.metric("Reduction", f"{red:.2f}%")

                out_suffix = safe_suffix_for_image(c_best.fmt)

                # If keeping extension, enforce original suffix (but only if it matches actual encoding)
                if keep_ext and orig_fmt == c_best.fmt and orig_suffix:
                    out_suffix = orig_suffix if orig_suffix.startswith(".") else ("." + orig_suffix)

                out_name = f"compressed_{up.name.rsplit('.', 1)[0]}{out_suffix}"
                mime = guess_mime(out_name)

                st.image(
                    Image.open(io.BytesIO(out_final)).convert("RGB"),
                    caption=f"Compressed ({len(out_final)/1024:.1f} KB, -{red:.2f}%) | Download below",
                    width="stretch",
                )

                st.download_button(
                    "Download compressed image",
                    data=out_final,
                    file_name=out_name,
                    mime=mime,
                )

        except Exception as e:
            show_exception(e, cfg)


# -----------------------------------------------------------------------------


def image_app(cfg):
    """Alias for legacy entrypoints."""
    return render_image_app(cfg)
