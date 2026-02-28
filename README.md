# 🔍 ClearSight v4 — Evidence-Grade CCTV Enhancement

A fully in-browser CCTV image enhancement system implementing the complete
pipeline from the ClearSight v4 Kaggle notebook — no GPU or backend required.

---

## Project Structure

```
clearsight-v4/
├── index.html                  ← Main HTML entry point
├── server.py                   ← Local dev server (Python stdlib only)
├── README.md
└── static/
    ├── css/
    │   └── main.css            ← All styles (variables, layout, components)
    └── js/
        ├── pipeline.js         ← Image processing algorithms (pure JS)
        ├── ui.js               ← UI controllers (Logger, Slider, Toggles…)
        └── app.js              ← App entry point, wires everything together
```

---

## Quick Start

### Option A — Python server (recommended)
```bash
cd clearsight-v4
python server.py
# Open http://localhost:8080
```

### Option B — Any static server
```bash
# Node.js
npx serve .

# Python (alternative port)
python -m http.server 3000

# VS Code: use Live Server extension
```

### Option C — Open directly
Double-click `index.html`.  
*(Some browsers restrict local file access — use Option A if images don't load.)*

---

## Enhancement Pipeline

The JavaScript pipeline in `static/js/pipeline.js` mirrors the notebook exactly:

| Step | Module | Description |
|------|--------|-------------|
| 0 | **Gatekeeper** | Laplacian blur score, brightness, fog score, motion |
| 1 | **DegradationClassifier** | FOG → NIGHT → BLUR → RAIN → CLEAR |
| 2 | **TemporalFusionSR** | CLAHE (clipLimit=3.0, tileGrid=8×8) |
| 3 | **DomainEnhancement** | Mode-specific processing (see below) |
| 4 | **ESRGANEnhancer** | Canvas bicubic 4× upscale + post-sharpen |
| 5 | **FidelityGuard** | White balance, contrast stretch, micro-detail |
| 6 | **EvidenceLogger** | JSON metadata export with case ID |

### Enhancement Modes

| Mode | Algorithm |
|------|-----------|
| **NIGHT** | Gamma lift (γ=0.55) → CLAHE → saturation boost → denoise → sharpen |
| **FOG** | Dark channel prior (AOD-Net style) → contrast stretch → sharpen |
| **BLUR** | Double-pass unsharp mask (σ=1.5, a=2.0 then σ=0.8, a=1.5) |
| **RAIN** | Vertical-streak suppression (Sobel-guided blur blend) → sharpen |
| **CLEAR** | Mild unsharp mask + contrast stretch |

---

## Configuration Options

| Setting | Options | Default |
|---------|---------|---------|
| Mode | AUTO / NIGHT / FOG / BLUR / RAIN / CLEAR | AUTO |
| Scale | 1× / 2× / 4× | 4× |
| Depth | Full Pipeline / Fast (SR only) | Full |
| Quality | Evidence-grade / High / Balanced | Evidence-grade |
| CLAHE | On / Off | On |
| Denoising | On / Off | On |
| Fidelity Guard | On / Off | On |

---

## Metrics Reported

- **PSNR (dB)** ↑ — Peak Signal-to-Noise Ratio
- **SSIM** ↑ — Structural Similarity Index
- **Fidelity Score** — 1 − normalised mean absolute difference
- **Degradation Type** — Auto-detected class

---

## Evidence Report

Click **EXPORT EVIDENCE REPORT** to download a JSON file containing:
```json
{
  "system": "ClearSight v4",
  "case_id": "case_1234567890",
  "timestamp": "2026-02-28T...",
  "pipeline": ["Gatekeeper", "DegradationClassifier", ...],
  "settings": { "mode": "auto", "scale": "4", ... },
  "degradation_type": "NIGHT",
  "metrics": { "psnr_db": 8.87, "ssim": 0.5171, "fidelity_score": 0.823 },
  "source_file": { "name": "frame.jpg", "size_kb": 142 }
}
```

---

## Browser Compatibility

| Browser | Status |
|---------|--------|
| Chrome 90+ | ✅ Full support |
| Firefox 88+ | ✅ Full support |
| Safari 15+ | ✅ Full support |
| Edge 90+ | ✅ Full support |

Requires: Canvas 2D API, FileReader API, CSS Custom Properties.

---

## Notebook Reference

Based on **ClearSight v4** Kaggle notebook using:
- `RealESRGAN` (RRDBNet, 23 blocks, 64 features, 4× scale)
- `basicsr` + `facexlib` + `gfpgan`
- `AOD-Net` for dehazing
- `CLAHE` via OpenCV (`clipLimit=3.0`, `tileGridSize=(8,8)`)
- `LPIPS` perceptual metric (AlexNet backbone)
- `EasyOCR` for CCPD plate recognition
- Datasets: LOL (night) · RESIDE (fog) · REDS (blur+SR) · CCPD2019 (plates)
