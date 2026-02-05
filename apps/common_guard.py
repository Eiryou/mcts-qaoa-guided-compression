#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
apps/common_guard.py

Shared production-ish safety guards for all apps:

- Upload size cap (all formats)
- Global concurrency semaphore (avoid CPU/RAM runaway on Render)
- Basic rate limit (per client-id/session)
- Detailed error shown only when NEO_DEBUG=1

NOTE:
This is not a security product. It's a pragmatic guardrail for public demos.
"""

from __future__ import annotations

import os
import time
import hashlib
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import streamlit as st

# Branding (shown near disclaimer)
DEVELOPER_NAME = str(os.getenv("DEVELOPER_NAME", "Hideyoshi Murakami")).strip() or "Hideyoshi Murakami"
X_HANDLE = str(os.getenv("X_HANDLE", "@nagisa7654321")).strip() or "@nagisa7654321"



def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip() in ("1", "true", "True", "yes", "YES")


@dataclass
class GuardConfig:
    # Upload hard limits (bytes)
    # Local-first default. Override with MAX_UPLOAD_MB if needed.
    # WARNING: Very large uploads can exhaust RAM/temporary disk.
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "500"))
    # Concurrency
    max_concurrency: int = int(os.getenv("MAX_CONCURRENCY", "1"))
    # Rate limit (simple token bucket per client)
    rate_window_sec: int = int(os.getenv("RATE_WINDOW_SEC", "60"))
    max_requests_per_window: int = int(os.getenv("MAX_REQ_PER_WINDOW", "12"))
    # Debug
    neo_debug: bool = _env_flag("NEO_DEBUG", "0")


# Global semaphore shared across modules/process threads (single worker typical on Render)
_SEM = threading.Semaphore(1)


def init_global_semaphore(max_concurrency: int) -> None:
    """
    Initialize global semaphore. Call once at app start.
    Streamlit reruns often, so we keep it simple.
    """
    global _SEM
    if max_concurrency < 1:
        max_concurrency = 1
    _SEM = threading.Semaphore(max_concurrency)


def _get_client_hint() -> str:
    """
    Best-effort client identifier. Streamlit does not expose IP reliably.
    We combine session + user agent-ish info when available.
    """
    # Streamlit session id (stable enough within a browser tab)
    sid = getattr(st.runtime.scriptrunner, "get_script_run_ctx", lambda: None)()
    sid_str = ""
    try:
        sid_str = sid.session_id if sid else ""
    except Exception:
        sid_str = ""

    # Headers are not always available; guard for failures
    ua = ""
    try:
        headers = st.context.headers  # Streamlit >=1.32
        ua = headers.get("User-Agent", "") if headers else ""
    except Exception:
        ua = ""

    raw = f"{sid_str}|{ua}"
    if not raw.strip("|"):
        raw = "anonymous"
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()[:16]


def check_upload_size(uploaded_file, cfg: GuardConfig) -> Tuple[bool, str]:
    """
    Return (ok, message). Uses uploaded_file.size when available.
    """
    if uploaded_file is None:
        return False, "No file."
    size = getattr(uploaded_file, "size", None)
    if size is None:
        # fallback: read bytes (avoid if possible)
        try:
            pos = uploaded_file.tell()
            data = uploaded_file.read()
            uploaded_file.seek(pos)
            size = len(data)
        except Exception:
            size = 0

    max_bytes = int(cfg.max_upload_mb) * 1024 * 1024
    if size > max_bytes:
        return False, f"Upload too large: {size/1024/1024:.2f} MB (limit {cfg.max_upload_mb} MB)."
    return True, ""


def _get_rate_bucket(cfg: GuardConfig):
    key = "_rate_bucket"
    if key not in st.session_state:
        st.session_state[key] = {}
    bucket = st.session_state[key]
    return bucket


def rate_limit_allow(cfg: GuardConfig) -> bool:
    """
    Simple per-client fixed window counter.
    """
    bucket = _get_rate_bucket(cfg)
    cid = _get_client_hint()
    now = time.time()

    win = cfg.rate_window_sec
    maxn = cfg.max_requests_per_window

    rec = bucket.get(cid)
    if rec is None:
        bucket[cid] = {"t0": now, "n": 1}
        return True

    if now - rec["t0"] >= win:
        rec["t0"] = now
        rec["n"] = 1
        return True

    if rec["n"] >= maxn:
        return False

    rec["n"] += 1
    return True


class ConcurrencyGuard:
    """
    Context manager to limit concurrent heavy jobs.
    """

    def __init__(self, cfg: GuardConfig):
        self.cfg = cfg
        self.acquired = False

    def __enter__(self):
        # Acquire semaphore
        self.acquired = _SEM.acquire(timeout=30)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.acquired:
            _SEM.release()


def guard_banner() -> None:
    """
    Display a prominent disclaimer banner (reusable).
    Keep it visible but not duplicated across pages.
    """
    st.markdown(
        f"**Developer:** {DEVELOPER_NAME} / **X:** [{X_HANDLE}](https://x.com/{X_HANDLE.lstrip('@')})"
    )
    st.warning(
        "⚠️ Disclaimer: Research/demo tool. Results may be degraded/corrupted; always keep backups. Do not upload sensitive data to public instances.",
        icon="⚠️"
    )
    with st.expander("Disclaimer (details) / 免責（詳細）"):
        st.write(
            """**English**
- This app is provided for research/demo purposes only.
- Outputs may lose quality, become unreadable, or change metadata.
- You are responsible for backups and for complying with any licenses/rights of the input content.
- Do not upload secrets or personal data to public deployments.

**日本語**
- 本アプリは研究・デモ目的で提供されます。
- 出力は品質低下・破損・メタデータ変更などが起こり得ます。必ずバックアップを保持してください。
- 入力コンテンツの権利・ライセンス遵守は利用者の責任です。
- 公開デプロイ（Render等）には機密情報や個人情報をアップロードしないでください。
"""
        )

def run_guard_or_stop(cfg: GuardConfig, uploaded_file) -> None:
    """
    Apply guards; if violated, show message and stop.
    """
    ok, msg = check_upload_size(uploaded_file, cfg)
    if not ok:
        st.error(msg)
        st.stop()

    if not rate_limit_allow(cfg):
        st.error("Rate limit exceeded. Please wait a bit and try again.")
        st.stop()


def show_exception(e: Exception, cfg: GuardConfig) -> None:
    """
    Show exception safely. Detailed only when NEO_DEBUG=1.
    """
    if cfg.neo_debug:
        st.exception(e)
    else:
        st.error("An error occurred. Enable NEO_DEBUG=1 to see details.")


# -----------------------------------------------------------------------------
# UI guard helper: prevent duplicate disclaimer rendering across tabs/apps
# -----------------------------------------------------------------------------
_DISCLAIMER_RENDERED = False

def set_disclaimer_rendered():
    global _DISCLAIMER_RENDERED
    _DISCLAIMER_RENDERED = True

def is_disclaimer_rendered() -> bool:
    return bool(_DISCLAIMER_RENDERED)
