#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
apps/pdf_app.py

PDF Optimizer (Probe → Final) - lightweight variant
- Probe: recompress a small sample of embedded images to rank settings
- Final: apply best setting to all images and save once

This is a pragmatic demo; PDF integrity/archival constraints (PDF/A, signatures) are not guaranteed.
"""

from __future__ import annotations

import io
import math
import random
import subprocess
import tempfile
import shutil
from dataclasses import dataclass
from typing import List, Tuple, Optional

import streamlit as st
import pikepdf
from pikepdf import Pdf, PdfImage, Name, Stream

from PIL import Image
import numpy as np
import zlib

from apps.common_guard import GuardConfig, run_guard_or_stop, ConcurrencyGuard, show_exception
from apps.utils import guess_mime


@dataclass
class ImgPolicy:
    fmt: str = "jpeg"
    quality: int = 65
    subsampling: int = 2  # 0=4:4:4, 1=4:2:2, 2=4:2:0
    downscale_max_px: int = 0
    gray: bool = False  # max side for embedded images; 0=keep


# --- Quantum-inspired search helpers (classical simulation) --------------------
# These are *inspired* by quantum annealing / interference, but run on CPU only.
try:
    from skimage.metrics import structural_similarity as _ssim
except Exception:  # pragma: no cover
    _ssim = None


def _fast_ssim_rgb(a: Image.Image, b: Image.Image, max_side: int = 256) -> float:
    """Fast SSIM on downscaled RGB images. Returns 0..1. Falls back to MSE->pseudo score."""
    a = a.convert("RGB")
    b = b.convert("RGB")
    # resize to same
    if a.size != b.size:
        b = b.resize(a.size, Image.BILINEAR)
    # downscale for speed
    m = max(a.size)
    if m > max_side:
        a = _resize_max_side(a, max_side)
        b = b.resize(a.size, Image.BILINEAR)
    if _ssim is None:
        arr_a = np.asarray(a).astype(np.float32) / 255.0
        arr_b = np.asarray(b).astype(np.float32) / 255.0
        mse = float(np.mean((arr_a - arr_b) ** 2))
        return float(max(0.0, 1.0 - math.sqrt(mse)))
    arr_a = np.asarray(a)
    arr_b = np.asarray(b)
    # channel-wise SSIM, then mean
    scores = []
    for c in range(3):
        scores.append(float(_ssim(arr_a[:, :, c], arr_b[:, :, c], data_range=255)))
    return float(max(0.0, min(1.0, sum(scores) / 3.0)))


def _probe_policy_on_images(pdf_bytes: bytes, pol: ImgPolicy, max_images: int = 5) -> Tuple[float, float, int]:
    """Return (avg_ssim, est_ratio, n_images_used) for a policy by sampling embedded images."""
    used = 0
    ssim_sum = 0.0
    orig_sum = 0
    new_sum = 0
    with pikepdf.open(io.BytesIO(pdf_bytes)) as pdf:
        for _, _, raw in _iter_pdf_images(pdf):
            im = _pil_from_pdfimage(raw)
            if im is None:
                continue
            used += 1
            orig_stream = bytes(raw.read_bytes())
            orig_sum += len(orig_stream)
            # apply same transforms as final (without writing back)
            pim = im
            if pol.gray:
                pim = pim.convert("L").convert("RGB")
            pim = _resize_max_side(pim, pol.downscale_max_px)
            enc, _ = _encode(pim, pol.fmt, pol.quality)
            new_sum += len(enc)
            ssim_sum += _fast_ssim_rgb(im, pim)
            if used >= max_images:
                break
    if used == 0:
        return 1.0, 1.0, 0
    avg_ssim = ssim_sum / used
    est_ratio = (new_sum / orig_sum) if orig_sum > 0 else 1.0
    return float(avg_ssim), float(est_ratio), used


@dataclass
class _QNode:
    action: Optional[Tuple[str, int]] = None  # (dim_name, choice_idx)
    parent: Optional["_QNode"] = None
    children: dict = None
    visits: int = 0
    value_sum: float = 0.0
    amp: complex = 0j  # interference accumulator

    def __post_init__(self):
        if self.children is None:
            self.children = {}

    @property
    def q(self) -> float:
        return self.value_sum / max(1, self.visits)


def _softmax(xs: List[float]) -> List[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps) + 1e-12
    return [e / s for e in exps]


def _entropy(ps: List[float]) -> float:
    e = 0.0
    for p in ps:
        p = max(1e-12, float(p))
        e -= p * math.log(p)
    return e


def _quantum_mcts_search(
    pdf_bytes: bytes,
    *,
    n_iters: int = 80,
    c_ucb: float = 1.2,
    tunnel_p: float = 0.08,
    phase_k: float = 0.35,
    max_probe_images: int = 5,
    ssim_floor: float = 0.93,
    seed: int = 0,
) -> ImgPolicy:
    """Search ImgPolicy with quantum-inspired exploration/interference (CPU simulation)."""
    rng = random.Random(seed)

    # Discrete action dimensions (tune for stronger PDF compression)
    dims = [
        ("quality", [40, 50, 60, 65, 70, 75, 80]),
        ("downscale", [0, 2200, 1800, 1400, 1100, 900]),
        ("subsampling", [2, 1, 0]),
        ("gray", [0, 1]),
    ]

    def rollout(partial: dict) -> Tuple[float, dict]:
        # Superposition: sample multiple heuristic weight-sets; prefer low-entropy (stable) winners
        # Base weights (energy/Hamiltonian)
        base = {"w_size": 1.0, "w_dist": 2.2, "w_time": 0.15}
        weight_sets = []
        for _ in range(4):
            w = dict(base)
            w["w_size"] *= rng.uniform(0.85, 1.25)
            w["w_dist"] *= rng.uniform(0.85, 1.35)
            w["w_time"] *= rng.uniform(0.7, 1.3)
            weight_sets.append(w)

        # complete policy
        d = dict(partial)
        for name, choices in dims:
            if name not in d:
                d[name] = rng.choice(choices)
        pol = ImgPolicy(
            fmt="jpeg",
            quality=int(d["quality"]),
            subsampling=int(d["subsampling"]),
            downscale_max_px=int(d["downscale"]),
            gray=bool(d["gray"]),
        )

        t0 = __import__("time").time()
        avg_ssim, est_ratio, used = _probe_policy_on_images(pdf_bytes, pol, max_images=max_probe_images)
        dt = __import__("time").time() - t0

        # energy per weight-set
        energies = []
        for w in weight_sets:
            dist = max(0.0, 1.0 - avg_ssim)
            # Hamiltonian: bitcost(size) + distortion + time + constraint penalty
            H = w["w_size"] * est_ratio + w["w_dist"] * dist + w["w_time"] * dt
            if avg_ssim < ssim_floor and used > 0:
                H += 10.0 * (ssim_floor - avg_ssim)  # hard-ish constraint
            energies.append(float(H))

        # collapse: choose lowest-energy configuration, but penalize high-entropy (unstable) weight response
        ps = _softmax([-e for e in energies])
        ent = _entropy(ps) / math.log(len(ps))  # normalized 0..1
        H_eff = min(energies) + 0.6 * ent
        reward = -H_eff
        return reward, d

    def select_child(node: _QNode, total_visits: int, depth: int) -> _QNode:
        # Quantum tunnel: occasionally jump to a random child to escape local minima
        if node.children and rng.random() < tunnel_p * math.exp(-depth / 6.0):
            return rng.choice(list(node.children.values()))

        best = None
        best_score = -1e18
        for k, ch in node.children.items():
            n = max(1, ch.visits)
            ucb = ch.q + c_ucb * math.sqrt(math.log(total_visits + 1) / n)
            inter = phase_k * (abs(ch.amp) / n)
            score = ucb + inter
            if score > best_score:
                best_score = score
                best = ch
        return best

    root = _QNode(action=None, parent=None)

    for it in range(n_iters):
        node = root
        partial = {}
        depth = 0
        # SELECTION + EXPANSION over dims
        while True:
            if depth >= len(dims):
                break
            dim_name, choices = dims[depth]
            # expand if not fully expanded
            if len(node.children) < len(choices):
                # pick an unexpanded choice
                unexp = [i for i in range(len(choices)) if i not in node.children]
                ci = rng.choice(unexp)
                child = _QNode(action=(dim_name, ci), parent=node)
                node.children[ci] = child
                node = child
                partial[dim_name] = choices[ci]
                depth += 1
                break
            # otherwise select best child
            node = select_child(node, root.visits + 1, depth)
            dim_name, ci = node.action
            partial[dim_name] = choices[ci]
            depth += 1

        # SIMULATION
        reward, completed = rollout(partial)

        # BACKPROP + phase interference
        # assign a deterministic-ish phase from path + iteration
        phi = (hash(str(sorted(completed.items())) + str(it)) % 360) * math.pi / 180.0
        amp_delta = reward * complex(math.cos(phi), math.sin(phi))

        cur = node
        while cur is not None:
            cur.visits += 1
            cur.value_sum += reward
            cur.amp += amp_delta
            cur = cur.parent

    # derive best policy from most visited / best Q among root's expanded paths
    best_partial = {}
    node = root
    for depth, (dim_name, choices) in enumerate(dims):
        if not node.children:
            break
        # pick child with best q
        best_child = max(node.children.values(), key=lambda ch: (ch.q, ch.visits))
        _, ci = best_child.action
        best_partial[dim_name] = choices[ci]
        node = best_child

    return ImgPolicy(
        fmt="jpeg",
        quality=int(best_partial.get("quality", 65)),
        subsampling=int(best_partial.get("subsampling", 2)),
        downscale_max_px=int(best_partial.get("downscale", 0)),
        gray=bool(best_partial.get("gray", 0)),
    )


def _iter_pdf_images(pdf: Pdf):
    for page in pdf.pages:
        try:
            for name, raw in page.images.items():
                yield page, name, raw
        except Exception:
            continue


def _pil_from_pdfimage(raw) -> Optional[Image.Image]:
    try:
        pim = PdfImage(raw)
        im = pim.as_pil_image()
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        else:
            im = im.convert("RGB")
        return im
    except Exception:
        return None


def _resize_max_side(im: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return im
    w, h = im.size
    m = max(w, h)
    if m <= max_side:
        return im
    scale = max_side / float(m)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return im.resize((nw, nh), Image.LANCZOS)


def _encode(im: Image.Image, fmt: str, quality: int) -> Tuple[bytes, str]:
    out = io.BytesIO()
    fmt = fmt.lower()
    if fmt == "jpeg":
        im.save(out, format="JPEG", quality=int(quality), optimize=True, progressive=True)
        return out.getvalue(), "DCTDecode"
    if fmt == "webp":
        im.save(out, format="WEBP", quality=int(quality), method=6)
        # In PDF, we will embed it as a Flate stream? PikePDF can't directly set WebP as XObject reliably.
        # So we fallback to JPEG for maximum compatibility.
        return out.getvalue(), "DCTDecode"
    # fallback
    im.save(out, format="JPEG", quality=int(quality), optimize=True, progressive=True)
    return out.getvalue(), "DCTDecode"


def _encode_jpeg(im: Image.Image, quality: int, subsampling: int = 2, progressive: bool = True) -> bytes:
    """Encode PIL image to JPEG bytes."""
    try:
        bio = io.BytesIO()
        save_kwargs = dict(format="JPEG", quality=int(quality), optimize=True)
        # Pillow supports 'subsampling' for JPEG
        try:
            save_kwargs["subsampling"] = int(subsampling)
        except Exception:
            pass
        if progressive:
            save_kwargs["progressive"] = True
        im.save(bio, **save_kwargs)
        return bio.getvalue()
    except Exception:
        return b""

def _is_text_like(im: Image.Image) -> bool:
    """Heuristic classifier: returns True for line-art / text scans."""
    try:
        # Small grayscale thumbnail
        g = im.convert("L")
        g = g.resize((128, 128), Image.BILINEAR)
        a = np.asarray(g, dtype=np.float32) / 255.0
        # Simple edge magnitude (Sobel-ish)
        dx = np.abs(a[:, 1:] - a[:, :-1])
        dy = np.abs(a[1:, :] - a[:-1, :])
        edge = float(dx.mean() + dy.mean())
        # Contrast
        contrast = float(a.std())
        # Text-like tends to have higher edge density with moderate/low std
        return (edge > 0.08 and contrast < 0.25) or (edge > 0.11)
    except Exception:
        return False

def _make_flate_image_stream(pdf: Pdf, im: Image.Image, bits: int = 8):
    """Create a FlateDecode PDF image XObject stream from a grayscale PIL image."""
    try:
        if bits == 1:
            g = im.convert("L")
            arr = np.asarray(g, dtype=np.uint8)
            # threshold at mid
            bw = (arr > 127).astype(np.uint8)
            # pack bits, MSB first per byte
            packed = np.packbits(bw, axis=1, bitorder="big")
            raw = packed.tobytes()
            bpc = 1
        else:
            g = im.convert("L")
            raw = np.asarray(g, dtype=np.uint8).tobytes()
            bpc = 8

        w, h = im.size
        comp = zlib.compress(raw, 9)
        st = Stream(pdf, comp)
        st["/Type"] = Name("/XObject")
        st["/Subtype"] = Name("/Image")
        st["/Width"] = int(w)
        st["/Height"] = int(h)
        st["/ColorSpace"] = Name("/DeviceGray")
        st["/BitsPerComponent"] = int(bpc)
        st["/Filter"] = Name("/FlateDecode")
        return st
    except Exception:
        return None

def _apply_policy_to_image(pdf: Pdf, page, name: str, raw, pol: ImgPolicy) -> bool:
    """Replace one XObject image with a recompressed version.

    Notes:
      - We skip soft masks / image masks to avoid breaking transparency.
      - For text-like images, we prefer 1-bit or Flate (lossless-ish) to keep edges.
      - Otherwise we use JPEG (DCTDecode) with optional downscale.
    """
    try:
        obj = raw.get_object()
        # Avoid breaking transparency/masks
        try:
            if "/SMask" in obj or "/Mask" in obj:
                return False
        except Exception:
            pass

        pimg = PdfImage(obj)
        im = pimg.as_pil_image()
        if im is None:
            return False

        # Optional downscale
        im2 = im
        if pol.downscale_max_px and pol.downscale_max_px > 0:
            w, h = im2.size
            m = max(w, h)
            if m > pol.downscale_max_px:
                s = pol.downscale_max_px / float(m)
                nw = max(1, int(w * s))
                nh = max(1, int(h * s))
                im2 = im2.resize((nw, nh), Image.LANCZOS)

        # Heuristic: text/line-art
        text_like = _is_text_like(im2)

        # Decide output mode
        if pol.gray:
            try:
                im2 = im2.convert("L")
            except Exception:
                pass

        if text_like:
            # 1-bit (bilevel) if it looks close to mono; else 8-bit gray
            try:
                im_l = im2.convert("L")
            except Exception:
                im_l = im2.convert("L") if im2.mode != "L" else im2

            # quick unique-level check on thumbnail
            try:
                thumb = im_l.resize((96, 96), Image.BILINEAR)
                arr = np.asarray(thumb)
                uniq = len(np.unique(arr))
            except Exception:
                uniq = 256

            if uniq <= 32:
                st = _make_flate_image_stream(pdf, im_l, bits=1)
            else:
                st = _make_flate_image_stream(pdf, im_l, bits=8)

            if st is None:
                return False

            page.images[name] = st
            return True

        # Photo-ish: JPEG
        jbytes = _encode_jpeg(im2, quality=int(pol.quality), subsampling=int(pol.subsampling), progressive=True)
        if not jbytes:
            return False

        # Build a PDF Image XObject stream (DCTDecode)
        if im2.mode != "RGB":
            try:
                im_rgb = im2.convert("RGB")
            except Exception:
                im_rgb = im2
        else:
            im_rgb = im2

        w, h = im_rgb.size
        st = Stream(pdf, jbytes)
        st["/Type"] = Name("/XObject")
        st["/Subtype"] = Name("/Image")
        st["/Width"] = int(w)
        st["/Height"] = int(h)
        st["/ColorSpace"] = Name("/DeviceRGB")
        st["/BitsPerComponent"] = 8
        st["/Filter"] = Name("/DCTDecode")

        page.images[name] = st
        return True
    except Exception:
        return False

def _strip_metadata(pdf: Pdf) -> None:
    try:
        if pdf.docinfo:
            pdf.docinfo.clear()
    except Exception:
        pass
    try:
        if "/Metadata" in pdf.Root:
            del pdf.Root["/Metadata"]
    except Exception:
        pass


def _save_pdf_bytes(pdf: Pdf) -> bytes:
    out = io.BytesIO()
    pdf.save(out, linearize=False, compress_streams=True, object_stream_mode=pikepdf.ObjectStreamMode.generate)
    return out.getvalue()



def _try_ghostscript_pdfwrite(pdf_bytes: bytes) -> Optional[bytes]:
    """Optional stronger PDF compression via Ghostscript (pdfwrite). Returns bytes or None."""
    gs = shutil.which("gs") or shutil.which("gswin64c") or shutil.which("gswin32c")
    if not gs:
        return None
    # A small set of presets; we choose the smallest output.
    presets = ["/screen", "/ebook", "/printer", "/prepress"]
    best = None
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in.pdf")
        with open(in_path, "wb") as f:
            f.write(pdf_bytes)
        for ps in presets:
            out_path = os.path.join(td, f"out_{ps.strip('/')}.pdf")
            cmd = [
                gs,
                "-sDEVICE=pdfwrite",
                "-dCompatibilityLevel=1.4",
                "-dNOPAUSE",
                "-dQUIET",
                "-dBATCH",
                "-dDetectDuplicateImages=true",
                "-dCompressFonts=true",
                "-dSubsetFonts=true",
                f"-dPDFSETTINGS={ps}",
                f"-sOutputFile={out_path}",
                in_path,
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                b = open(out_path, "rb").read()
                if (best is None) or (len(b) < len(best)):
                    best = b
            except Exception:
                continue
    return best


def render_pdf_app(cfg: GuardConfig) -> None:
    st.header("PDF Optimizer (Probe → Final)")
    st.caption("Sample embedded images in probe; apply best setting to whole PDF in final.")
    up = st.file_uploader("Upload a PDF", type=["pdf"])
    if not up:
        return
    run_guard_or_stop(cfg, up)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        probe_images = st.slider("Probe images", 1, 10, 3)
    with col2:
        downscale = st.selectbox("Embedded image downscale max_side", [0, 1024, 768, 512, 384], index=2)
    with col3:
        q_min, q_max = st.slider("JPEG quality range", 5, 95, (25, 75))
    with col4:
        trials = st.slider("Probe trials", 5, 120, 40)
        ssim_floor = st.slider("SSIM floor (probe, embedded-image)", 0.80, 0.99, 0.93, 0.01)

    keep_text = st.checkbox("Keep text/vector intact (avoid rasterization)", value=True, disabled=True)

    if st.button("Optimize (Probe → Final)"):
        try:
            with ConcurrencyGuard(cfg):
                pdf_bytes = up.getvalue()
                base_size = len(pdf_bytes)
                base_size2 = base_size  # fallback; overwritten after strip+save

                with Pdf.open(io.BytesIO(pdf_bytes)) as pdf:
                    _strip_metadata(pdf)
                    # Baseline after metadata strip + re-save (reduces noise from metadata/stream repack)
                    base_saved = _save_pdf_bytes(pdf)
                    base_size2 = len(base_saved)

                    imgs = list(_iter_pdf_images(pdf))
                    if not imgs:
                        # still can compress streams
                        out_b = _save_pdf_bytes(pdf)
                        st.info("No embedded images found. Applied stream compression + metadata strip.")
                        st.download_button("Download optimized PDF", data=out_b,
                                           file_name=f"compressed_{up.name}",
                                           mime=guess_mime("x.pdf"))
                        st.write(f"Size: {base_size/1024:.1f} KB → {len(out_b)/1024:.1f} KB")
                        
                        red = (1.0 - (len(out_b)/float(base_size))) * 100.0 if base_size else 0.0
                        st.metric("Reduction", f"{red:.2f}%")
                        return

                    sample = imgs[:probe_images]
                    # Quantum-inspired two-phase search (Probe -> Final)
                    # Probe uses embedded-image sampling (fast) and a Hamiltonian energy model:
                    #   H = w_size*bitcost + w_dist*distortion + w_time*time + penalties
                    # plus 'tunneling' (random jumps) and 'phase interference' (complex accumulator).
                    prog = st.progress(0)
                    n_iters = int(min(300, max(120, trials * 70)))
                    best_pol = _quantum_mcts_search(
                        base_saved,
                        n_iters=n_iters,
                        tunnel_p=0.12,
                        phase_k=0.40,
                        max_probe_images=max(3, probe_images),
                        ssim_floor=float(ssim_floor),
                        seed=int(seed),
                    )
                    # Probe metrics for reporting
                    t0 = time.time()
                    p_ssim, p_ratio, used = _probe_policy_on_images(base_saved, best_pol, max_images=max(3, probe_images))
                    dt = time.time() - t0
                    st.success(
                        f"Probe best: SSIM~{p_ssim:.4f} (on {used} images), est.img-ratio~{p_ratio:.3f}, policy={best_pol}"
                    )

                    # Final apply to all images once
                    with Pdf.open(io.BytesIO(base_saved)) as pdf_f:
                        _strip_metadata(pdf_f)
                        changed = 0
                        for page, name, raw in _iter_pdf_images(pdf_f):
                            if _apply_policy_to_image(pdf_f, page, name, raw, best_pol):
                                changed += 1
                        out_final = _save_pdf_bytes(pdf_f)

                    out_name = f"compressed_{up.name.rsplit('.',1)[0]}.pdf"
                    st.download_button("Download optimized PDF", data=out_final, file_name=out_name, mime="application/pdf")
                    st.write(f"Images processed: {changed}")
                    st.write(f"Size: {base_size/1024:.1f} KB → {len(out_final)/1024:.1f} KB  (Δ {(len(out_final)-base_size)/1024:.1f} KB)")
        
                    red = (1.0 - (len(out_final)/float(base_size))) * 100.0 if base_size else 0.0
                    st.metric("Reduction", f"{red:.2f}%")
        except Exception as e:
            show_exception(e, cfg)
