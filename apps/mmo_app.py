#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
apps/mmo_app.py

MCTS Multimedia Optimizer (Probe → Final) - lightweight variant
- Probe on first N seconds to rank candidate encodes
- Final encodes full media once with best candidate
- Preserves playable extensions/containers where possible

NEW (Local Quantum Computing Fusion):
- Build discrete candidate grid (audio bitrate or video CRF)
- Probe-measure each discrete candidate (size, time, and video SSIM via ffmpeg filter)
- Run QAOA(p=1) statevector simulation to generate top-K promising candidates
- Mix QAOA top-K + (optional) quantum-inspired sampling + random trials
"""

from __future__ import annotations

import os
import tempfile
import subprocess
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import streamlit as st

from apps.quantum_inspired import PhaseInterferenceSampler
from apps.common_guard import GuardConfig, run_guard_or_stop, ConcurrencyGuard, show_exception
from apps.utils import guess_mime, safe_suffix_for_media


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class MediaCandidate:
    kind: str  # 'audio'|'video'
    vcodec: Optional[str]
    acodec: Optional[str]
    crf: Optional[int]
    abr_kbps: Optional[int]  # audio bitrate
    preset: str


# =============================================================================
# FFmpeg helpers
# =============================================================================

def _has_ffmpeg() -> bool:
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        return r.returncode == 0
    except Exception:
        return False


def _run_ffmpeg(cmd: List[str]) -> Tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    # Keep tail for debugging
    return p.returncode, (p.stderr or "")[-6000:]


def _run_ffprobe_duration(path: Path) -> Optional[float]:
    """
    Returns duration seconds (float) using ffprobe, or None if fails.
    """
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path)
        ]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            return None
        s = (p.stdout or "").strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _probe_encode(inp: Path, outp: Path, cand: MediaCandidate, probe_sec: int) -> Tuple[bool, int, float, str]:
    """
    Return (ok, out_size_bytes, elapsed_sec, log_tail)
    """
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(inp)]
    if probe_sec > 0:
        cmd += ["-t", str(probe_sec)]

    if cand.kind == "audio":
        cmd += ["-vn"]
        cmd += ["-c:a", cand.acodec or "libmp3lame"]
        if cand.abr_kbps:
            cmd += ["-b:a", f"{cand.abr_kbps}k"]
    else:
        # video
        cmd += ["-c:v", cand.vcodec or "libx264", "-preset", cand.preset]
        if cand.crf is not None:
            cmd += ["-crf", str(cand.crf)]
        # keep audio reasonably (fixed)
        cmd += ["-c:a", "aac", "-b:a", "128k"]

    cmd += [str(outp)]

    t0 = time.time()
    rc, log = _run_ffmpeg(cmd)
    t1 = time.time()

    if rc != 0 or not outp.exists():
        return False, 0, (t1 - t0), log
    return True, outp.stat().st_size, (t1 - t0), log


def _final_encode(inp: Path, outp: Path, cand: MediaCandidate) -> Tuple[bool, float, str]:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(inp)]

    if cand.kind == "audio":
        cmd += ["-vn", "-c:a", cand.acodec or "libmp3lame"]
        if cand.abr_kbps:
            cmd += ["-b:a", f"{cand.abr_kbps}k"]
    else:
        cmd += ["-c:v", cand.vcodec or "libx264", "-preset", cand.preset]
        if cand.crf is not None:
            cmd += ["-crf", str(cand.crf)]
        cmd += ["-c:a", "aac", "-b:a", "128k"]

    cmd += [str(outp)]

    t0 = time.time()
    rc, log = _run_ffmpeg(cmd)
    t1 = time.time()
    return (rc == 0 and outp.exists()), (t1 - t0), log


def _video_ssim_ffmpeg(ref_path: Path, dist_path: Path, probe_sec: int) -> Optional[float]:
    """
    Compute SSIM between reference and distorted video using ffmpeg's ssim filter.
    We compare only the probe segment length to keep it cheap.
    Returns float SSIM in [0,1] or None if fails.

    NOTE: This is a practical metric, not perfect (alignment issues may exist).
    """
    # We use -t probe_sec on BOTH inputs for comparable lengths.
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "info",
        "-t", str(probe_sec), "-i", str(ref_path),
        "-t", str(probe_sec), "-i", str(dist_path),
        "-lavfi", "ssim",
        "-f", "null", "-"
    ]
    rc, log = _run_ffmpeg(cmd)
    if rc != 0:
        return None

    # Parse "All:0.987654" from the tail
    # Typical lines: "SSIM Y:... U:... V:... All:0.987654 (xx.xx)"
    marker = "All:"
    idx = log.rfind(marker)
    if idx < 0:
        return None
    tail = log[idx: idx + 80]
    # Extract number after "All:"
    try:
        s = tail.split("All:")[1].strip()
        num = ""
        for ch in s:
            if ch.isdigit() or ch in ".-eE":
                num += ch
            else:
                break
        val = float(num)
        if 0.0 <= val <= 1.0:
            return val
        return None
    except Exception:
        return None


# =============================================================================
# QAOA p=1 statevector simulator (same idea as image)
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
    qubit: 0 is LSB.
    """
    dim = 1 << n_qubits
    state = state.reshape([2] * n_qubits)

    axes = list(range(n_qubits))
    axes[0], axes[qubit] = axes[qubit], axes[0]
    state_t = np.transpose(state, axes).reshape(2, -1)

    state_t = gate @ state_t

    state_t = state_t.reshape([2] + [2] * (n_qubits - 1))
    inv_axes = np.argsort(axes)
    out = np.transpose(state_t, inv_axes).reshape(dim)
    return out


def _qaoa_p1_distribution(energies: np.ndarray,
                          gamma_grid: np.ndarray,
                          beta_grid: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    """
    Simulate QAOA p=1 for diagonal energies[z].
    Choose gamma,beta by minimizing expected energy.
    Return (probs, best_gamma, best_beta, best_expE).
    """
    energies = np.asarray(energies, dtype=np.float64)
    dim = energies.shape[0]
    n_qubits = int(np.log2(dim))

    base = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)

    best_expE = float("inf")
    best_gamma = float(gamma_grid[0])
    best_beta = float(beta_grid[0])
    best_probs = np.ones(dim, dtype=np.float64) / dim

    for gamma in gamma_grid:
        phase = np.exp(-1j * gamma * energies).astype(np.complex128)

        for beta in beta_grid:
            stv = base * phase
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


def _make_levels_int(lo: int, hi: int, levels: int) -> List[int]:
    if levels <= 1:
        return [int(lo)]
    xs = np.linspace(lo, hi, num=int(levels))
    ys = [int(round(x)) for x in xs]
    out = []
    for y in ys:
        if y not in out:
            out.append(y)
    return out


def _qaoa_topk_for_audio(
    inp: Path,
    suffix: str,
    acodec: str,
    abr_levels: List[int],
    probe_sec: int,
    prefer_input_ext: bool,
    alpha_size: float,
    gamma_steps: int,
    beta_steps: int,
    top_k: int,
) -> Tuple[List[Tuple[MediaCandidate, str]], Dict[str, float]]:
    """
    Build energies over discrete abr levels using probe encode size+time.
    QAOA returns top-K abr choices to prioritize.
    """
    # combos = abr_levels
    combos = list(abr_levels)

    # cap combos for speed
    max_dim = 64
    if len(combos) > max_dim:
        rng = random.Random(1337)
        rng.shuffle(combos)
        combos = combos[:max_dim]

    n = len(combos)
    n_qubits = int(np.ceil(np.log2(max(1, n))))
    dim = 1 << n_qubits

    energies = np.full(dim, 1e6, dtype=np.float64)
    skipped = 0
    measured = 0

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for idx, abr in enumerate(combos):
            cand = MediaCandidate(kind="audio", vcodec=None, acodec=acodec, crf=None, abr_kbps=int(abr), preset="medium")
            out_suf = safe_suffix_for_media("audio", "mp3" if acodec == "libmp3lame" else "aac", suffix if prefer_input_ext else "")

            outp = td / f"qaoa_probe_{idx}{out_suf}"
            ok, sz, elapsed, _log = _probe_encode(inp, outp, cand, probe_sec)
            if not ok:
                skipped += 1
                continue

            size_kb = sz / 1024.0
            # Energy: size + small time penalty (local)
            E = (alpha_size * size_kb) + (elapsed * 5.0)
            energies[idx] = float(E)
            measured += 1

    gamma_grid = np.linspace(0.0, 2.5, num=max(3, int(gamma_steps))).astype(np.float64)
    beta_grid = np.linspace(0.0, np.pi / 2.0, num=max(3, int(beta_steps))).astype(np.float64)
    probs, gbest, bbest, expE = _qaoa_p1_distribution(energies, gamma_grid, beta_grid)

    order = np.argsort(-probs)
    picks: List[Tuple[MediaCandidate, str]] = []
    seen = set()

    for j in order:
        j = int(j)
        if j >= n:
            continue
        abr = int(combos[j])
        if abr in seen:
            continue
        seen.add(abr)

        cand = MediaCandidate(kind="audio", vcodec=None, acodec=acodec, crf=None, abr_kbps=abr, preset="medium")
        out_suf = safe_suffix_for_media("audio", "mp3" if acodec == "libmp3lame" else "aac", suffix if prefer_input_ext else "")
        picks.append((cand, out_suf))

        if len(picks) >= int(top_k):
            break

    stats = dict(
        qaoa_kind=1.0,
        qaoa_ncombos=float(n),
        qaoa_dim=float(dim),
        qaoa_measured=float(measured),
        qaoa_skipped=float(skipped),
        qaoa_best_gamma=float(gbest),
        qaoa_best_beta=float(bbest),
        qaoa_best_expE=float(expE),
    )
    return picks, stats


def _qaoa_topk_for_video(
    inp: Path,
    suffix: str,
    vcodec: str,
    preset: str,
    crf_levels: List[int],
    probe_sec: int,
    prefer_input_ext: bool,
    alpha_size: float,
    beta_dist: float,
    min_ssim: float,
    gamma_steps: int,
    beta_steps: int,
    top_k: int,
) -> Tuple[List[Tuple[MediaCandidate, str]], Dict[str, float]]:
    """
    Build energies over discrete CRF levels using probe encode + SSIM(ref, dist) via ffmpeg filter.
    Energy: alpha*size_kb + beta*(1-ssim) + time penalty + floor penalty.
    """
    combos = list(crf_levels)

    max_dim = 64
    if len(combos) > max_dim:
        rng = random.Random(1337)
        rng.shuffle(combos)
        combos = combos[:max_dim]

    n = len(combos)
    n_qubits = int(np.ceil(np.log2(max(1, n))))
    dim = 1 << n_qubits

    energies = np.full(dim, 1e6, dtype=np.float64)
    skipped = 0
    measured = 0

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Reference probe segment file (lossless-ish): we can just cut original without re-encode
        # But container/copy may fail; easiest: create a reference mp4 using libx264 CRF 0? too heavy.
        # Practical: use original as reference directly with -t on inputs in SSIM call.
        # We'll compute SSIM between inp and encoded probe.

        for idx, crf in enumerate(combos):
            cand = MediaCandidate(kind="video", vcodec=vcodec, acodec="aac", crf=int(crf), abr_kbps=None, preset=preset)
            out_suf = safe_suffix_for_media("video", vcodec, suffix if prefer_input_ext else "")

            outp = td / f"qaoa_probe_{idx}{out_suf}"
            ok, sz, elapsed, _log = _probe_encode(inp, outp, cand, probe_sec)
            if not ok:
                skipped += 1
                continue

            ssim_val = _video_ssim_ffmpeg(inp, outp, probe_sec)
            if ssim_val is None:
                # if we can't compute SSIM, treat as bad
                ssim_val = 0.0

            size_kb = sz / 1024.0
            dist = (1.0 - float(ssim_val))

            penalty = 0.0
            if float(ssim_val) < float(min_ssim):
                penalty = 80.0 + (float(min_ssim) - float(ssim_val)) * 300.0

            E = (alpha_size * size_kb) + (beta_dist * dist) + (elapsed * 10.0) + penalty
            energies[idx] = float(E)
            measured += 1

    gamma_grid = np.linspace(0.0, 2.5, num=max(3, int(gamma_steps))).astype(np.float64)
    beta_grid = np.linspace(0.0, np.pi / 2.0, num=max(3, int(beta_steps))).astype(np.float64)
    probs, gbest, bbest, expE = _qaoa_p1_distribution(energies, gamma_grid, beta_grid)

    order = np.argsort(-probs)
    picks: List[Tuple[MediaCandidate, str]] = []
    seen = set()

    for j in order:
        j = int(j)
        if j >= n:
            continue
        crf = int(combos[j])
        if crf in seen:
            continue
        seen.add(crf)

        cand = MediaCandidate(kind="video", vcodec=vcodec, acodec="aac", crf=crf, abr_kbps=None, preset=preset)
        out_suf = safe_suffix_for_media("video", vcodec, suffix if prefer_input_ext else "")
        picks.append((cand, out_suf))

        if len(picks) >= int(top_k):
            break

    stats = dict(
        qaoa_kind=2.0,
        qaoa_ncombos=float(n),
        qaoa_dim=float(dim),
        qaoa_measured=float(measured),
        qaoa_skipped=float(skipped),
        qaoa_best_gamma=float(gbest),
        qaoa_best_beta=float(bbest),
        qaoa_best_expE=float(expE),
    )
    return picks, stats


# =============================================================================
# Main Streamlit App
# =============================================================================

def render_media_app(cfg: GuardConfig) -> None:
    st.header("Media Optimizer (Probe → Final)")
    st.caption("Optimizes audio/video with a short probe search, then encodes full media once.")
    if not _has_ffmpeg():
        st.error("ffmpeg not found. Install ffmpeg for local run, or ensure your deployment image provides it.")
        return

    up = st.file_uploader("Upload audio/video", type=["mp3", "wav", "m4a", "aac", "opus", "mp4", "mov", "mkv", "webm"])
    if not up:
        return
    run_guard_or_stop(cfg, up)

    name = up.name
    suffix = "." + name.split(".")[-1].lower()
    kind = "audio" if suffix in (".mp3", ".wav", ".m4a", ".aac", ".opus") else "video"

    col1, col2, col3 = st.columns(3)
    with col1:
        probe_sec = st.slider("Probe seconds", 2, 30, 8)
    with col2:
        trials = st.slider("Probe trials (random)", 5, 80, 24)
    with col3:
        prefer_input_ext = st.checkbox("Prefer input extension/container", value=True)

    # Quantum-inspired sampler (optional)
    st.subheader("Quantum-inspired (phase interference)")
    use_qi = st.checkbox("Enable quantum-inspired bias (PhaseInterferenceSampler)", value=True)
    qi_seed = st.number_input("QI seed", min_value=0, max_value=2**31 - 1, value=1337, step=1)
    qi_sampler = PhaseInterferenceSampler(seed=int(qi_seed))

    # Quantum computing (QAOA)
    st.subheader("Quantum Computing Fusion (QAOA circuit simulation)")
    use_qaoa = st.checkbox("Enable QAOA(p=1) candidate generator", value=True)
    qaoa_topk = st.slider("QAOA top-K candidates", 2, 24, 10, step=1)
    qaoa_levels = st.slider("Discrete levels for QAOA grid", 3, 20, 8, step=1)
    qaoa_gamma_steps = st.slider("QAOA gamma grid steps", 5, 31, 11, step=2)
    qaoa_beta_steps = st.slider("QAOA beta grid steps", 5, 31, 11, step=2)

    # Energy weights
    colw1, colw2 = st.columns(2)
    with colw1:
        alpha_size = st.slider("Energy weight: size (alpha)", 0.1, 5.0, 1.0, step=0.1)
    with colw2:
        beta_dist = st.slider("Energy weight: distortion (beta, video only)", 50.0, 3000.0, 900.0, step=50.0)

    if kind == "audio":
        abr_min, abr_max = st.slider("Audio bitrate range (kbps)", 32, 320, (64, 192))
        acodec = "libmp3lame" if suffix == ".mp3" else "aac"
        if st.checkbox("Force MP3 output when input is MP3", value=(suffix == ".mp3")):
            acodec = "libmp3lame"
        st.info(f"Audio codec candidate: {acodec}")
    else:
        vcodec = st.selectbox("Video codec", ["libx264", "libx265"], index=0)
        crf_min, crf_max = st.slider("CRF range (lower=better)", 18, 40, (24, 34))
        preset = st.selectbox("Preset", ["ultrafast", "veryfast", "fast", "medium"], index=1)
        min_ssim = st.slider("Min SSIM (video probe)", 0.60, 0.99, 0.92, step=0.01)

    if st.button("Optimize (Probe → Final)"):
        try:
            with ConcurrencyGuard(cfg):
                with tempfile.TemporaryDirectory() as td:
                    td = Path(td)
                    inp = td / f"input{suffix}"
                    inp.write_bytes(up.getvalue())

                    base_size = inp.stat().st_size

                    # ---------------------------
                    # Build candidate list
                    # ---------------------------
                    qaoa_stats: Dict[str, float] = {}
                    qaoa_candidates: List[Tuple[MediaCandidate, str]] = []

                    if use_qaoa:
                        if kind == "audio":
                            abr_levels = _make_levels_int(int(abr_min), int(abr_max), int(qaoa_levels))
                            qaoa_candidates, qaoa_stats = _qaoa_topk_for_audio(
                                inp=inp,
                                suffix=suffix,
                                acodec=acodec,
                                abr_levels=abr_levels,
                                probe_sec=int(probe_sec),
                                prefer_input_ext=bool(prefer_input_ext),
                                alpha_size=float(alpha_size),
                                gamma_steps=int(qaoa_gamma_steps),
                                beta_steps=int(qaoa_beta_steps),
                                top_k=int(qaoa_topk),
                            )
                        else:
                            crf_levels = _make_levels_int(int(crf_min), int(crf_max), int(qaoa_levels))
                            qaoa_candidates, qaoa_stats = _qaoa_topk_for_video(
                                inp=inp,
                                suffix=suffix,
                                vcodec=vcodec,
                                preset=preset,
                                crf_levels=crf_levels,
                                probe_sec=int(probe_sec),
                                prefer_input_ext=bool(prefer_input_ext),
                                alpha_size=float(alpha_size),
                                beta_dist=float(beta_dist),
                                min_ssim=float(min_ssim),
                                gamma_steps=int(qaoa_gamma_steps),
                                beta_steps=int(qaoa_beta_steps),
                                top_k=int(qaoa_topk),
                            )

                    # Random trials (with optional quantum-inspired bias)
                    rng = random.Random(1337)
                    random_candidates: List[Tuple[MediaCandidate, str]] = []

                    for _ in range(int(trials)):
                        if kind == "audio":
                            abr = rng.randint(int(abr_min), int(abr_max))
                            # QI bias: update amplitude to prefer lower bitrates mildly but still explore
                            if use_qi:
                                # "energy" lower is better: lower abr => lower energy
                                # normalize to ~[0,1]
                                e = (abr - int(abr_min)) / max(1.0, float(int(abr_max) - int(abr_min)))
                                qi_sampler.update(abr, float(e))
                                abr = int(qi_sampler.sample(list(range(int(abr_min), int(abr_max) + 1))))
                            cand = MediaCandidate(kind="audio", vcodec=None, acodec=acodec, crf=None, abr_kbps=int(abr), preset="medium")
                            out_suf = safe_suffix_for_media("audio", "mp3" if acodec == "libmp3lame" else "aac", suffix if prefer_input_ext else "")
                        else:
                            crf = rng.randint(int(crf_min), int(crf_max))
                            if use_qi:
                                # prefer mid/high CRF for size but keep exploration
                                # energy lower is better -> energy = (crf - crf_min)/(crf_max-crf_min)
                                e = (crf - int(crf_min)) / max(1.0, float(int(crf_max) - int(crf_min)))
                                qi_sampler.update(crf, float(e))
                                crf = int(qi_sampler.sample(list(range(int(crf_min), int(crf_max) + 1))))
                            cand = MediaCandidate(kind="video", vcodec=vcodec, acodec="aac", crf=int(crf), abr_kbps=None, preset=preset)
                            out_suf = safe_suffix_for_media("video", vcodec, suffix if prefer_input_ext else "")

                        random_candidates.append((cand, out_suf))

                    # Merge: QAOA first (priority), then random
                    all_candidates = list(qaoa_candidates) + list(random_candidates)

                    # Dedup by parameter signature
                    uniq: List[Tuple[MediaCandidate, str]] = []
                    seen = set()
                    for cand, out_suf in all_candidates:
                        key = (
                            cand.kind,
                            cand.vcodec or "",
                            cand.acodec or "",
                            int(cand.crf) if cand.crf is not None else -1,
                            int(cand.abr_kbps) if cand.abr_kbps is not None else -1,
                            cand.preset or "",
                            out_suf,
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        uniq.append((cand, out_suf))
                    all_candidates = uniq

                    # ---------------------------
                    # Probe search
                    # ---------------------------
                    best = None
                    best_E = float("inf")
                    best_probe_size = 10**18
                    prog = st.progress(0)

                    for i, (cand, out_suf) in enumerate(all_candidates, 1):
                        outp = td / f"probe_{i}{out_suf}"
                        ok, sz, elapsed, _log = _probe_encode(inp, outp, cand, int(probe_sec))
                        if not ok:
                            prog.progress(i / len(all_candidates))
                            continue

                        size_kb = sz / 1024.0
                        time_pen = elapsed * (10.0 if kind == "video" else 5.0)

                        if kind == "audio":
                            # No perceptual metric here (simple and robust).
                            E = (float(alpha_size) * size_kb) + time_pen
                        else:
                            ssim_val = _video_ssim_ffmpeg(inp, outp, int(probe_sec))
                            if ssim_val is None:
                                ssim_val = 0.0
                            dist = (1.0 - float(ssim_val))
                            penalty = 0.0
                            if float(ssim_val) < float(min_ssim):
                                penalty = 80.0 + (float(min_ssim) - float(ssim_val)) * 300.0
                            E = (float(alpha_size) * size_kb) + (float(beta_dist) * dist) + time_pen + penalty

                        if E < best_E:
                            best_E = E
                            best_probe_size = sz
                            best = (cand, out_suf)

                        prog.progress(i / len(all_candidates))

                    if best is None:
                        st.error("Probe failed (no successful candidate).")
                        if use_qaoa and qaoa_stats:
                            st.info(f"QAOA stats: {qaoa_stats}")
                        return

                    cand_best, out_suf = best

                    st.success(
                        f"Probe best: probe_size≈{best_probe_size/1024:.1f} KB (base {base_size/1024:.1f} KB) | "
                        f"E≈{best_E:.3f}"
                    )
                    if use_qaoa and qaoa_stats:
                        st.info(
                            f"QAOA stats: combos={int(qaoa_stats.get('qaoa_ncombos',0))} "
                            f"measured={int(qaoa_stats.get('qaoa_measured',0))} "
                            f"skipped={int(qaoa_stats.get('qaoa_skipped',0))} "
                            f"best_gamma={qaoa_stats.get('qaoa_best_gamma',0):.3f} "
                            f"best_beta={qaoa_stats.get('qaoa_best_beta',0):.3f} "
                            f"best_expE={qaoa_stats.get('qaoa_best_expE',0):.3f}"
                        )

                    # ---------------------------
                    # Final encode once
                    # ---------------------------
                    outp_final = td / f"compressed{out_suf}"
                    ok, elapsed, log = _final_encode(inp, outp_final, cand_best)
                    if not ok:
                        st.error("Final encode failed.")
                        if cfg.neo_debug:
                            st.code(log)
                        return

                    out_bytes = outp_final.read_bytes()
                    out_name = f"compressed_{Path(name).stem}{out_suf}"
                    mime = guess_mime(out_name)

                    st.write(f"Size: {base_size/1024:.1f} KB → {len(out_bytes)/1024:.1f} KB")
                    red = (1.0 - (len(out_bytes) / float(base_size))) * 100.0 if base_size else 0.0
                    st.metric("Reduction", f"{red:.2f}%")

                    # Preview (browser may not support all containers)
                    if kind == "audio":
                        st.audio(out_bytes, format=mime)
                    else:
                        st.video(out_bytes, format=mime)

                    st.download_button("Download compressed media", data=out_bytes, file_name=out_name, mime=mime)

        except Exception as e:
            show_exception(e, cfg)


# Backward-compatible alias
def mmo_app(cfg: GuardConfig) -> None:
    return render_media_app(cfg)
