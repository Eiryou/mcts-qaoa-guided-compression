#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import mimetypes
from pathlib import Path

def guess_mime(filename: str) -> str:
    mt, _ = mimetypes.guess_type(filename)
    return mt or "application/octet-stream"

def safe_suffix_for_image(fmt: str) -> str:
    fmt = (fmt or "").lower()
    if fmt in ("jpeg", "jpg"):
        return ".jpg"
    if fmt == "png":
        return ".png"
    if fmt == "webp":
        return ".webp"
    if fmt == "avif":
        return ".avif"
    return ".bin"

def safe_suffix_for_media(kind: str, codec: str, input_suffix: str) -> str:
    """
    Decide container suffix.
    kind: 'audio'|'video'
    codec: requested codec name (e.g., 'mp3','aac','opus','h264','hevc','av1')
    input_suffix: like '.mp3' '.mp4'
    """
    s = (input_suffix or "").lower()
    c = (codec or "").lower()

    # Prefer input container when feasible
    if kind == "audio":
        if c in ("mp3",) and s == ".mp3":
            return ".mp3"
        if c in ("aac", "libfdk_aac"):
            return ".m4a"
        if c in ("opus",):
            return ".opus"
        # default
        return ".mp3" if s == ".mp3" else ".m4a"

    # video
    if kind == "video":
        if s in (".mp4", ".m4v") and c in ("h264","libx264","hevc","h265","libx265","av1","libaom-av1","svtav1"):
            return ".mp4"
        if s in (".webm",) and c in ("vp9","libvpx-vp9","av1","libaom-av1","svtav1"):
            return ".webm"
        # default
        return ".mp4"
