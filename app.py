#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Murakami Compressor Suite - Streamlit router
Developer: Hideyoshi Murakami / X: @nagisa7654321
License: MIT

Modes:
- Image (MCTS Image Optimizer style)
- Media (MCTS Multimedia Optimizer style)

Note:
  PDF compression is intentionally NOT bundled here.
  It is best maintained as a separate project because robust PDF rewriting
  (fonts, forms, signatures, PDF/A, etc.) requires a different pipeline and
  heavier dependencies.
"""

import os
import streamlit as st

from apps.common_guard import GuardConfig, init_global_semaphore, guard_banner
# NOTE: Individual sub-apps expose render_* functions.
# We intentionally avoid importing non-existent legacy symbols like `image_app`.

from apps.image_app import render_image_app
from apps.mmo_app import render_media_app


def main():
    st.set_page_config(page_title="Murakami Compressor Suite", layout="wide")

    cfg = GuardConfig()
    init_global_semaphore(cfg.max_concurrency)

    st.title("Murakami Compressor Suite")
    st.caption("Search-based compression across PDF / Image / Media  |  Developer: Hideyoshi Murakami  |  X: @nagisa7654321")

    # Prominent disclaimer near the identity block (requested)
    guard_banner()

    mode = st.sidebar.radio("Mode", ["Image", "Media"], index=0)

    if mode == "Image":
        render_image_app(cfg)
    else:
        render_media_app(cfg)


if __name__ == "__main__":
    main()
