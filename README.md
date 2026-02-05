# Murakami Compressor Suite

**Search-Based Compression** for Images + Audio/Video  
with **Quantum-Inspired Sampling** and optional **QAOA(p=1) Fusion** (classical simulation)

Combining MCTS, heuristics, and quantum computing

**Author / Developer:** Hideyoshi Murakami  
**X:** @nagisa7654321
## X(Twitter): https://x.com/nagisa7654321

## DOI (Zenodo)
https://doi.org/10.5281/zenodo.18489982

## Qiita
https://qiita.com/Hideyoshi_Murakami/items/38f80882013d6c22b44f

> **Disclaimer**
> This project does **not** claim quantum speedup or quantum hardware advantage.
> “Quantum-inspired” and QAOA here are **classical heuristics / simulations** used to improve exploration and candidate prioritization.

## Quick Start (Local)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## What’s inside
- `apps/image_app.py` — Image Probe→Final optimizer (SSIM-based)
- `apps/mmo_app.py` — Media Probe→Final optimizer (SSIM probe + optional VMAF refine + QAOA fusion)
- `TECHNICAL.md` — deep technical explanation with QAOA equations and diagrams
- `technical.txt` — plain-text technical explanation
- `TECHNICAL.md — QAOA(p=1): energy landscape C(z) → probability P(z) → top-K candidate prioritization
## License
Licensed under the **Apache License 2.0**.

**Attribution required:** Please retain the copyright notice and the `NOTICE` file
when redistributing or creating derivative works, as required by Apache-2.0.

**Optional request (non-binding):** If you redistribute a modified version, please notify
the author via **GitHub Issues** or **X (Twitter)**: **@nagisa7654321**.
