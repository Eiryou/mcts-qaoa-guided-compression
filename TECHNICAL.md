# TECHNICAL.md — Search-Based Compression with Quantum-Inspired + QAOA Fusion

## Attribution
- Author / Developer: Hideyoshi Murakami
- X (Twitter): @nagisa7654321

## 0) What “Quantum Fusion” means in this repo (important)
This project does **NOT** claim quantum speedup or quantum hardware advantage.

Instead, it **fuses quantum-algorithm structure** (QAOA) into a *classical* search pipeline:
- We probe-measure candidate encodes to build an **energy landscape** `C(z)`.
- We run a **QAOA(p=1) statevector simulation** to convert `C(z)` into a probability distribution `P(z)`.
- We try **top-K** candidates from `P(z)` first, then mix with random / quantum-inspired exploration.

In short: **QAOA is used as a candidate prioritizer for compression search.**

---

## 1) Problem framing: compression as an optimization-by-search task
Most compression workflows rely on presets (codec/CRF/bitrate/quality).  
This repo reframes compression as:

> Choose parameters (actions) that minimize an energy (cost), subject to quality constraints.

### 1.1 Action space (examples)
**Images**
- `fmt ∈ {jpeg, webp, avif, png}`
- `quality ∈ [q_min, q_max]` for lossy formats
- optional resize controls (`probe_max_side`, `final_max_side`)

**Media**
- Audio: `abr_kbps ∈ [min, max]` (bitrate search)
- Video: `vcodec ∈ {x264, x265}`, `preset`, `crf ∈ [min, max]`

---

## 2) Probe → Final: the core practical design
### 2.1 Why Probe → Final
Full encodes are expensive (especially video).  
So we split the process into:
- **Probe stage**: many cheap trials (downscaled image or first N seconds of media)
- **Final stage**: encode full content **exactly once** using the best candidate

This makes “search-based compression” feasible under real CPU/RAM constraints.

### 2.2 Scoring / energy examples
**Image score (maximize):**
- `score = SSIM(orig_probe, cand_probe)*1000  - size_kB  - time_penalty*50`
- plus a hard floor: reject if `SSIM < min_ssim`

**Media energy (minimize):**
- Audio (simple & robust):
  - `E = α * size_kB  +  time_penalty`
- Video (two-stage probe):
  - Stage-1 (cheap): `SSIM` via FFmpeg `ssim` filter
  - Stage-2 (expensive, only top-M): `VMAF` via FFmpeg `libvmaf` filter (if available)

Video energy examples:
- Stage-1: `E = α*size_kB + β*(1-SSIM) + time_penalty + floor_penalty`
- Stage-2: `E2 = α*size_kB + w*(100-VMAF)/100 + time_penalty + vmaf_penalty`

---

## 3) Quantum-Inspired Sampling (Phase Interference) — classical heuristic
`PhaseInterferenceSampler` maintains a complex amplitude `amp[k]` per discrete key `k`.

- Update step (lower energy ⇒ stronger amplitude magnitude):
  - `mag = exp(-max(0, energy))`
  - `amp[k] ← amp[k] + mag * exp(i * phase)`

- Sampling probability:
  - `P(k) ∝ |amp[k]|²`

**Effect**
- Biases selection toward “likely-good” actions (exploitation)
- Keeps randomness / interference-like behavior (exploration)

---

## 4) QAOA(p=1) Fusion — the “quantum algorithm structure” part
### 4.1 Mapping compression candidates to QAOA states
We discretize a parameter axis into N levels:
- Audio: bitrate levels (e.g., 8 values)
- Video: CRF levels (e.g., 8 values)

Each discrete candidate is an index `z` (bitstring) mapped to a basis state `|z⟩`.

We probe-measure each candidate to compute an **energy** (cost) `C(z)`:
- size + time (+ distortion for video + penalties)

This creates an **energy landscape** over the discrete candidate set.

### 4.2 QAOA operators and formulas
**Cost Hamiltonian (diagonal):**
- `H_C |z⟩ = C(z) |z⟩`

**Phase separator:**
- `U_C(γ) = exp(-i γ H_C)`
- `U_C(γ) |z⟩ = exp(-i γ C(z)) |z⟩`

**Mixer Hamiltonian (X-mixer):**
- `H_B = Σ_j X_j`

**Mixer operator:**
- `U_B(β) = exp(-i β H_B) = Π_j exp(-i β X_j)`

**Initial state (uniform superposition):**
- `|s⟩ = |+⟩^{⊗n} = (1/√2^n) Σ_z |z⟩`

**p=1 QAOA state:**
- `|ψ(γ,β)⟩ = U_B(β) U_C(γ) |s⟩`

**Measurement probability distribution:**
- `P(z; γ,β) = |⟨z|ψ(γ,β)⟩|²`

We select `(γ,β)` by a simple grid search that minimizes expected energy:
- `E_exp(γ,β) = Σ_z P(z;γ,β) C(z)`

### 4.3 How QAOA is fused into the search pipeline
1) Discretize candidates and probe-measure energies `C(z)`
2) Run QAOA(p=1) statevector simulation → obtain `P(z)`
3) Take **top-K** candidates by `P(z)` as “promising”
4) Merge with random / quantum-inspired exploration candidates
5) Run the Probe loop in that merged order
6) Final encode once with the best candidate found

---

## 5) One-page diagram: “Energy landscape → Probability distribution → Search”
```
                (Probe measurements)
          +--------------------------------+
          |  Candidate grid (discrete z)   |
          |  z0 z1 z2 ... zN               |
          +----------------+---------------+
                           |
                           v
                 +------------------+
                 |  Energy C(z)     |
                 |  (size,dist,time)|
                 +------------------+
                           |
                           |  QAOA(p=1) statevector sim
                           |  U_C(γ)=exp(-iγH_C)
                           |  U_B(β)=exp(-iβΣX_j)
                           v
                 +------------------+
                 |  Prob. P(z)      |
                 |  = |<z|ψ>|^2     |
                 +------------------+
                           |
                 top-K by P(z)  +  random/QI candidates
                           |
                           v
                 +---------------------------+
                 |   Probe Search Loop       |
                 |   Stage-1: SSIM for all   |
                 |   Stage-2: VMAF for top-M |
                 +---------------------------+
                           |
                           v
                 +---------------------------+
                 |   Final Encode (once)     |
                 |   full-res / full-length  |
                 +---------------------------+
```
