## ![](https://github-visitor-counter-tau.vercel.app/api?username=Eiryou)

# Murakami Compressor Suite
> **Compression is not a preset — it’s a search (Probe → Final).**

## 2MB -> 200KB (10%~90% reduction) without changing extension

## CPU usage: negligible / RAM usage: minimal 

My development test environment: Ryzen7 5700G/DDR4 64GB/GTX1070

**Search-based compression for Images / Audio / Video** with:
- **Probe → Final** (cheap probe search, **single** full encode)
- **Quality-aware constraints** (**SSIM** for images, **VMAF/SSIM** for video)
- **MCTS-ish search + heuristics**
- **Quantum-inspired sampling** (phase interference / |amp|²)
- Optional **QAOA(p=1)** fusion (**classical statevector simulation**) to generate a **proposal prior** (Top-K candidates)

**Author / Developer:** Hideyoshi Murakami  
**X:** @nagisa7654321 (https://x.com/nagisa7654321)

---

## Links
- **GitHub:** https://github.com/Eiryou/mcts-qaoa-guided-compression  
- **Zenodo (DOI):** https://doi.org/10.5281/zenodo.18489982  
- **Qiita:** https://qiita.com/Hideyoshi_Murakami/items/38f80882013d6c22b44f  

---

## Why this matters
Most compression tools require manual trial-and-error (codec, CRF/quality, bitrate, resize, presets…).  
This project treats compression as an **optimization/search problem**:

1) **Probe** cheaply measures many candidates (downscaled image / first N seconds of media)  
2) Builds an **energy landscape** `C(z)` from **size + distortion + time + penalties**  
3) Uses **QAOA(p=1)** (classical simulation) and/or **quantum-inspired sampling** to propose promising candidates  
4) Runs **Final** encoding **once** with the best candidate

---

## Disclaimer (Important)
This project does **not** claim **quantum speedup** or **quantum hardware advantage**.  
“Quantum-inspired” sampling and QAOA(p=1) are **classical heuristics / simulations** used to improve exploration and candidate prioritization.

---

## Quick Start (Local)

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

### Requirements
- **Python 3.10+** recommended  
- **ffmpeg** is required for audio/video optimization  
- For **VMAF**, your ffmpeg must include **libvmaf** (your build does)

---

## What’s inside
- `apps/image_app.py` — Image optimizer (**Probe → Final**, SSIM-based)  
- `apps/mmo_app.py` — Media optimizer (**Probe → Final**, SSIM probe + optional VMAF + optional QAOA fusion)  
- `TECHNICAL.md` — Deep technical explanation (QAOA equations + diagrams)  
- `technical.txt` — Plain-text technical explanation  

---

## How it works (one-line summary)
**QAOA(p=1) turns an energy landscape `C(z)` → a proposal distribution `P(z)` → prioritizes Top-K candidates** for the Probe stage.

If you want the full math & diagrams:
- See **TECHNICAL.md**

---

## Tips (practical)
- Start with **Probe seconds = 6–10** for video; shorter is faster but noisier  
- **VMAF is heavier** than SSIM → reduce trials, rely on **Top-K + a few random**  
- Keep **Final = once** (design principle)

---

## Contact
For comments, work, and collaborations, please contact us here and If you want to use it for commercial purposes, please contact me
## murakami3tech6compres9sion@gmail.com   

## Support / Feedback
If this project helped you, consider giving it a ⭐ — it directly supports further development and documentation.


---

## License
Licensed under the **Apache License 2.0**.

**Attribution required:** Please retain the copyright notice and the `NOTICE` file when redistributing or creating derivative works, as required by Apache-2.0.


