/**
 * pipeline.js
 * ClearSight v4 — Image Enhancement Pipeline
 *
 * Implements the full processing chain from the ClearSight v4 notebook:
 *   Gatekeeper → DegradationClassifier → TemporalFusionSR →
 *   DomainEnhancement (Night/Fog/Blur/Rain/Clear) → ESRGANEnhancer →
 *   FidelityGuard → EvidenceLogger
 *
 * All processing runs in-browser via Canvas 2D API.
 */

'use strict';

/* ═══════════════════════════════════════════════════════════
   GATEKEEPER — Analyzes raw frame quality metrics
   ═══════════════════════════════════════════════════════════ */
const Gatekeeper = {
  /**
   * Compute per-pixel luminance array from RGBA data.
   * @param {Uint8ClampedArray} data
   * @returns {Float32Array}
   */
  getLuminance(data) {
    const n = data.length / 4;
    const lum = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      lum[i] = 0.299 * data[i*4] + 0.587 * data[i*4+1] + 0.114 * data[i*4+2];
    }
    return lum;
  },

  /**
   * Laplacian variance — proxy for blur score.
   * Higher = sharper. Threshold from notebook: 120.0
   */
  blurScore(lum, W, H) {
    let sum = 0, sum2 = 0, count = 0;
    for (let y = 1; y < H-1; y++) {
      for (let x = 1; x < W-1; x++) {
        const c = lum[y*W+x];
        const lap = 4*c
          - lum[(y-1)*W+x] - lum[(y+1)*W+x]
          - lum[y*W+(x-1)] - lum[y*W+(x+1)];
        sum  += lap;
        sum2 += lap * lap;
        count++;
      }
    }
    const mean = sum / count;
    return sum2 / count - mean * mean;   // variance
  },

  /**
   * Mean brightness (0–255). Threshold: dark < 60
   */
  brightness(lum) {
    let s = 0;
    for (let i = 0; i < lum.length; i++) s += lum[i];
    return s / lum.length;
  },

  /**
   * Fog score — mean of per-pixel min(R,G,B).
   * High min-channel → low dark channel → foggy. Threshold: > 170
   */
  fogScore(data) {
    let s = 0;
    const n = data.length / 4;
    for (let i = 0; i < n; i++) {
      s += Math.min(data[i*4], data[i*4+1], data[i*4+2]);
    }
    return s / n;
  },

  /**
   * Run full gatekeeper analysis.
   * @returns {{ blurScore, brightness, fogScore, isDark, isBlurry, isFoggy, needsEnhancement }}
   */
  analyze(data, W, H, blurThreshold = 120, motionThreshold = 500) {
    const lum  = this.getLuminance(data);
    const bs   = this.blurScore(lum, W, H);
    const br   = this.brightness(lum);
    const fs   = this.fogScore(data);

    return {
      blurScore:        Math.round(bs * 100) / 100,
      brightness:       Math.round(br * 100) / 100,
      fogScore:         Math.round(fs * 100) / 100,
      isBlurry:         bs  < blurThreshold,
      isDark:           br  < 60,
      isFoggy:          fs  > 170,
      needsEnhancement: bs < blurThreshold || br < 60 || fs > 170,
    };
  },
};


/* ═══════════════════════════════════════════════════════════
   DEGRADATION CLASSIFIER
   ═══════════════════════════════════════════════════════════ */
const DegradationClassifier = {
  TYPES: { FOG: 'FOG', NIGHT: 'NIGHT', BLUR: 'BLUR', RAIN: 'RAIN', CLEAR: 'CLEAR' },

  classify(gate, data, W, H) {
    if (gate.isFoggy)  return this.TYPES.FOG;
    if (gate.isDark)   return this.TYPES.NIGHT;
    if (gate.isBlurry) return this.TYPES.BLUR;

    // Rain: dominant vertical edges via Sobel
    const lum = Gatekeeper.getLuminance(data);
    let sx = 0, sy = 0;
    for (let y = 1; y < H-1; y++) {
      for (let x = 1; x < W-1; x++) {
        const gx = lum[y*W+(x+1)] - lum[y*W+(x-1)];
        const gy = lum[(y+1)*W+x] - lum[(y-1)*W+x];
        sx += Math.abs(gx);
        sy += Math.abs(gy);
      }
    }
    return sy > sx * 1.8 ? this.TYPES.RAIN : this.TYPES.CLEAR;
  },
};


/* ═══════════════════════════════════════════════════════════
   CLAHE — Contrast Limited Adaptive Histogram Equalization
   Mirrors notebook: clipLimit=3.0, tileGridSize=(8,8)
   ═══════════════════════════════════════════════════════════ */
const CLAHE = {
  apply(data, W, H, tiles = 8, clipLimit = 3.0) {
    const result = new Uint8ClampedArray(data);
    const tW = Math.floor(W / tiles);
    const tH = Math.floor(H / tiles);

    for (let ty = 0; ty < tiles; ty++) {
      for (let tx = 0; tx < tiles; tx++) {
        const x0 = tx * tW, y0 = ty * tH;
        const x1 = tx === tiles-1 ? W : x0 + tW;
        const y1 = ty === tiles-1 ? H : y0 + tH;
        const area = (x1-x0) * (y1-y0);

        // Build luminance histogram
        const hist = new Float64Array(256);
        for (let y = y0; y < y1; y++) {
          for (let x = x0; x < x1; x++) {
            const i = (y*W+x) * 4;
            const l = Math.round(0.299*data[i] + 0.587*data[i+1] + 0.114*data[i+2]);
            hist[l]++;
          }
        }

        // Clip and redistribute
        const clip = Math.max(1, clipLimit * area / 256);
        let excess = 0;
        for (let v = 0; v < 256; v++) {
          if (hist[v] > clip) { excess += hist[v] - clip; hist[v] = clip; }
        }
        const add = excess / 256;
        for (let v = 0; v < 256; v++) hist[v] += add;

        // CDF
        const cdf = new Float64Array(256);
        cdf[0] = hist[0];
        for (let v = 1; v < 256; v++) cdf[v] = cdf[v-1] + hist[v];
        let cdfMin = 0;
        for (let v = 0; v < 256; v++) { if (cdf[v] > 0) { cdfMin = cdf[v]; break; } }

        // Mapping LUT
        const lut = new Uint8ClampedArray(256);
        for (let v = 0; v < 256; v++) {
          lut[v] = Math.round(((cdf[v] - cdfMin) / (area - cdfMin + 1)) * 255);
        }

        // Apply per-pixel with saturation preservation
        for (let y = y0; y < y1; y++) {
          for (let x = x0; x < x1; x++) {
            const i = (y*W+x) * 4;
            const r = data[i], g = data[i+1], b = data[i+2];
            const lum = Math.round(0.299*r + 0.587*g + 0.114*b);
            const newLum = lut[lum];
            const scale = lum > 0 ? newLum / lum : 1;
            result[i]   = Math.min(255, r * scale);
            result[i+1] = Math.min(255, g * scale);
            result[i+2] = Math.min(255, b * scale);
          }
        }
      }
    }
    return result;
  },
};


/* ═══════════════════════════════════════════════════════════
   GAUSSIAN BLUR — Used by unsharp mask & denoising
   ═══════════════════════════════════════════════════════════ */
const GaussianBlur = {
  /**
   * Separable box-blur approximation of Gaussian.
   * @param {Uint8ClampedArray} data
   * @param {number} W, H
   * @param {number} sigma
   */
  apply(data, W, H, sigma = 1.0) {
    const r = Math.max(1, Math.round(sigma * 2));
    const tmp = this._pass(data, W, H, r, true);
    return this._pass(tmp, W, H, r, false);
  },

  _pass(data, W, H, r, horizontal) {
    const result = new Uint8ClampedArray(data.length);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        let sr = 0, sg = 0, sb = 0, count = 0;
        for (let d = -r; d <= r; d++) {
          const nx = horizontal ? Math.max(0, Math.min(W-1, x+d)) : x;
          const ny = horizontal ? y : Math.max(0, Math.min(H-1, y+d));
          const i = (ny*W+nx) * 4;
          sr += data[i]; sg += data[i+1]; sb += data[i+2]; count++;
        }
        const oi = (y*W+x) * 4;
        result[oi]   = sr / count;
        result[oi+1] = sg / count;
        result[oi+2] = sb / count;
        result[oi+3] = data[oi+3];
      }
    }
    return result;
  },
};


/* ═══════════════════════════════════════════════════════════
   UNSHARP MASK — Core sharpening used throughout pipeline
   ═══════════════════════════════════════════════════════════ */
function unsharpMask(data, W, H, sigma = 1.0, amount = 1.5) {
  const blurred = GaussianBlur.apply(data, W, H, sigma);
  const result  = new Uint8ClampedArray(data.length);
  for (let i = 0; i < data.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      result[i+c] = Math.min(255, Math.max(0,
        data[i+c] + amount * (data[i+c] - blurred[i+c])
      ));
    }
    result[i+3] = data[i+3];
  }
  return result;
}


/* ═══════════════════════════════════════════════════════════
   DOMAIN ENHANCERS
   ═══════════════════════════════════════════════════════════ */

/** NIGHT: Gamma lift + CLAHE + saturation boost */
function enhanceNight(data, W, H) {
  // 1. Gamma correction (gamma < 1 brightens shadows)
  const gamma   = 0.55;
  const gammaLUT = new Uint8ClampedArray(256);
  for (let i = 0; i < 256; i++) {
    gammaLUT[i] = Math.round(255 * Math.pow(i / 255, gamma));
  }

  const lifted = new Uint8ClampedArray(data.length);
  for (let i = 0; i < data.length; i += 4) {
    lifted[i]   = gammaLUT[data[i]];
    lifted[i+1] = gammaLUT[data[i+1]];
    lifted[i+2] = gammaLUT[data[i+2]];
    lifted[i+3] = data[i+3];
  }

  // 2. CLAHE on lifted image
  const clahed = CLAHE.apply(lifted, W, H, 8, 3.0);

  // 3. Saturation boost (in-place)
  const result = new Uint8ClampedArray(clahed);
  const satBoost = 1.35;
  for (let i = 0; i < result.length; i += 4) {
    const r = clahed[i], g = clahed[i+1], b = clahed[i+2];
    const lum = 0.299*r + 0.587*g + 0.114*b;
    result[i]   = Math.min(255, Math.max(0, lum + (r-lum)*satBoost));
    result[i+1] = Math.min(255, Math.max(0, lum + (g-lum)*satBoost));
    result[i+2] = Math.min(255, Math.max(0, lum + (b-lum)*satBoost));
  }

  // 4. FastNlMeans-style denoise (light Gaussian)
  const denoised = GaussianBlur.apply(result, W, H, 0.8);

  // 5. Final sharpen
  return unsharpMask(denoised, W, H, 1.0, 1.8);
}

/** FOG: Dark-channel prior dehazing (AOD-Net style) */
function dehazeFog(data, W, H) {
  const n = data.length / 4;

  // Estimate atmospheric light from top-0.1% brightest pixels
  const brightness = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    brightness[i] = Math.max(data[i*4], data[i*4+1], data[i*4+2]);
  }
  const sorted = Float32Array.from(brightness).sort();
  const A = sorted[Math.floor(sorted.length * 0.999)] || 220;

  // Dark channel (min over small patch)
  const patchR = 7;
  const darkCh = new Float32Array(n);
  const W_ = W, H_ = H;
  for (let y = 0; y < H_; y++) {
    for (let x = 0; x < W_; x++) {
      let minV = 255;
      for (let dy = -patchR; dy <= patchR; dy++) {
        for (let dx = -patchR; dx <= patchR; dx++) {
          const ny = Math.max(0, Math.min(H_-1, y+dy));
          const nx = Math.max(0, Math.min(W_-1, x+dx));
          const i  = (ny*W_+nx) * 4;
          minV = Math.min(minV, data[i], data[i+1], data[i+2]);
        }
      }
      darkCh[y*W_+x] = minV;
    }
  }

  // Transmission estimate
  const omega = 0.95;
  const result = new Uint8ClampedArray(data);
  for (let i = 0; i < n; i++) {
    const t = Math.max(0.1, 1 - omega * darkCh[i] / A);
    result[i*4]   = Math.min(255, Math.max(0, (data[i*4]   - A) / t + A));
    result[i*4+1] = Math.min(255, Math.max(0, (data[i*4+1] - A) / t + A));
    result[i*4+2] = Math.min(255, Math.max(0, (data[i*4+2] - A) / t + A));
  }

  // Contrast boost + sharpen
  const stretched = stretchContrast(result);
  return unsharpMask(stretched, W, H, 0.8, 1.2);
}

/** BLUR: Wiener-style deconvolution approximation + unsharp mask */
function enhanceBlur(data, W, H) {
  // Multiple passes of increasing sharpness
  let d = unsharpMask(data, W, H, 1.5, 2.0);
  d = unsharpMask(d, W, H, 0.8, 1.5);
  d = stretchContrast(d);
  return d;
}

/** RAIN: Median approximation + vertical streak suppression */
function enhanceRain(data, W, H) {
  // Light Gaussian to suppress high-frequency streaks
  const blurred = GaussianBlur.apply(data, W, H, 1.2);

  // Blend: prefer blurred where vertical edges are very strong
  const lum = Gatekeeper.getLuminance(data);
  const result = new Uint8ClampedArray(data);
  for (let y = 1; y < H-1; y++) {
    for (let x = 0; x < W; x++) {
      const gy = Math.abs(lum[(y+1)*W+x] - lum[(y-1)*W+x]);
      if (gy > 40) {
        const i = (y*W+x) * 4;
        const alpha = Math.min(1, gy / 80);
        result[i]   = data[i]   * (1-alpha) + blurred[i]   * alpha;
        result[i+1] = data[i+1] * (1-alpha) + blurred[i+1] * alpha;
        result[i+2] = data[i+2] * (1-alpha) + blurred[i+2] * alpha;
      }
    }
  }
  return unsharpMask(result, W, H, 1.0, 1.3);
}

/** CLEAR: Standard mild sharpen + contrast stretch */
function enhanceClear(data, W, H) {
  let d = unsharpMask(data, W, H, 1.0, 1.2);
  return stretchContrast(d);
}


/* ═══════════════════════════════════════════════════════════
   FIDELITY GUARD — Contrast stretch + colour preservation
   ═══════════════════════════════════════════════════════════ */
function stretchContrast(data) {
  const result = new Uint8ClampedArray(data);
  for (let c = 0; c < 3; c++) {
    let mn = 255, mx = 0;
    for (let i = c; i < data.length; i += 4) {
      if (data[i] < mn) mn = data[i];
      if (data[i] > mx) mx = data[i];
    }
    const range = mx - mn || 1;
    for (let i = c; i < result.length; i += 4) {
      result[i] = Math.round((data[i] - mn) / range * 255);
    }
  }
  return result;
}

/** White balance correction (grey-world assumption) */
function whiteBalance(data) {
  let rSum = 0, gSum = 0, bSum = 0, n = data.length / 4;
  for (let i = 0; i < data.length; i += 4) {
    rSum += data[i]; gSum += data[i+1]; bSum += data[i+2];
  }
  const rMean = rSum/n, gMean = gSum/n, bMean = bSum/n;
  const mean = (rMean + gMean + bMean) / 3;
  const rScale = mean / (rMean || 1);
  const gScale = mean / (gMean || 1);
  const bScale = mean / (bMean || 1);
  const result = new Uint8ClampedArray(data);
  for (let i = 0; i < result.length; i += 4) {
    result[i]   = Math.min(255, result[i]   * rScale);
    result[i+1] = Math.min(255, result[i+1] * gScale);
    result[i+2] = Math.min(255, result[i+2] * bScale);
  }
  return result;
}

/** Micro-detail recovery: high-frequency edge blend */
function microDetailRecovery(original, enhanced, alpha = 0.15) {
  const result = new Uint8ClampedArray(enhanced);
  for (let i = 0; i < result.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      const detail = original[i+c] - enhanced[i+c];
      result[i+c] = Math.min(255, Math.max(0, enhanced[i+c] + detail * alpha));
    }
  }
  return result;
}


/* ═══════════════════════════════════════════════════════════
   SUPER-RESOLUTION UPSCALER
   Uses canvas bicubic interpolation + post-sharpen
   ═══════════════════════════════════════════════════════════ */
const ESRGANEnhancer = {
  /**
   * Upscale image by `scale` factor using high-quality canvas interpolation
   * then apply post-processing sharpening to simulate ESRGAN output.
   * @param {HTMLImageElement} img
   * @param {number} scale  2 or 4
   * @returns {HTMLCanvasElement}
   */
  enhance(img, scale, processedData = null, origW = 0, origH = 0) {
    const srcW = processedData ? origW : img.naturalWidth;
    const srcH = processedData ? origH : img.naturalHeight;
    const dstW = srcW * scale;
    const dstH = srcH * scale;

    const canvas = document.createElement('canvas');
    canvas.width  = dstW;
    canvas.height = dstH;
    const ctx = canvas.getContext('2d');
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    if (processedData) {
      // Draw processed pixel data via intermediate canvas
      const src = document.createElement('canvas');
      src.width = srcW; src.height = srcH;
      const sCtx = src.getContext('2d');
      const id = sCtx.createImageData(srcW, srcH);
      id.data.set(processedData);
      sCtx.putImageData(id, 0, 0);
      ctx.drawImage(src, 0, 0, dstW, dstH);
    } else {
      ctx.drawImage(img, 0, 0, dstW, dstH);
    }

    // Post-sharpen to simulate ESRGAN edge recovery
    let id = ctx.getImageData(0, 0, dstW, dstH);
    let data = unsharpMask(id.data, dstW, dstH, 1.2, 1.6);
    data = unsharpMask(data, dstW, dstH, 0.5, 0.8);  // fine detail pass
    id.data.set(data);
    ctx.putImageData(id, 0, 0);

    return canvas;
  },
};


/* ═══════════════════════════════════════════════════════════
   MAIN PIPELINE ORCHESTRATOR
   ═══════════════════════════════════════════════════════════ */
const Pipeline = {
  /**
   * Run the full ClearSight v4 enhancement pipeline.
   *
   * @param {HTMLImageElement} img        Source image element
   * @param {object}  opts                Enhancement options
   * @param {string}  opts.mode           'auto'|'night'|'fog'|'blur'|'rain'|'clear'
   * @param {number}  opts.scale          Upscale factor (1, 2, or 4)
   * @param {boolean} opts.clahe          Apply CLAHE
   * @param {boolean} opts.denoise        Apply denoising
   * @param {boolean} opts.fidelityGuard  Apply fidelity guard
   * @param {function} opts.onStep        Callback(stepIndex) for progress
   * @returns {{ canvas, metrics, degradationType }}
   */
  async run(img, opts) {
    const {
      mode = 'auto',
      scale = 4,
      clahe = true,
      denoise = true,
      fidelityGuard = true,
      onStep = () => {},
    } = opts;

    // ── Draw source to canvas
    const srcCanvas = document.createElement('canvas');
    srcCanvas.width  = img.naturalWidth;
    srcCanvas.height = img.naturalHeight;
    const ctx = srcCanvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    const W = srcCanvas.width, H = srcCanvas.height;
    let imageData = ctx.getImageData(0, 0, W, H);
    let data = new Uint8ClampedArray(imageData.data);

    // ── STEP 0: Gatekeeper
    onStep(0);
    await tick();
    const gate = Gatekeeper.analyze(data, W, H);

    // ── STEP 1: Degradation Classification
    onStep(1);
    await tick();
    let degradationType;
    if (mode === 'auto') {
      degradationType = DegradationClassifier.classify(gate, data, W, H);
    } else {
      degradationType = mode.toUpperCase();
    }

    // ── STEP 2: Temporal Fusion (simulated — single frame, use CLAHE as proxy)
    onStep(2);
    await tick();
    if (clahe) {
      data = CLAHE.apply(data, W, H, 8, 3.0);
    }

    // ── STEP 3: Domain Enhancement
    onStep(3);
    await tick();
    switch (degradationType) {
      case 'NIGHT': data = enhanceNight(data, W, H); break;
      case 'FOG':   data = dehazeFog(data, W, H);    break;
      case 'BLUR':  data = enhanceBlur(data, W, H);  break;
      case 'RAIN':  data = enhanceRain(data, W, H);  break;
      default:      data = enhanceClear(data, W, H); break;
    }

    // ── STEP 4: Denoising pass (before upscale)
    if (denoise) {
      data = GaussianBlur.apply(data, W, H, 0.6);
    }

    // ── STEP 4: ESRGAN Upscale
    onStep(4);
    await tick();
    const upscaled = ESRGANEnhancer.enhance(img, scale, data, W, H);
    const uW = upscaled.width, uH = upscaled.height;
    const uCtx = upscaled.getContext('2d');
    let uId = uCtx.getImageData(0, 0, uW, uH);
    let uData = new Uint8ClampedArray(uId.data);

    // ── STEP 5: Fidelity Guard
    onStep(5);
    await tick();
    if (fidelityGuard) {
      uData = whiteBalance(uData);
      uData = stretchContrast(uData);
      // Final polish
      uData = unsharpMask(uData, uW, uH, 0.6, 0.7);
    }

    uId.data.set(uData);
    uCtx.putImageData(uId, 0, 0);

    // ── STEP 6: Evidence Log (metadata)
    onStep(6);
    await tick();

    // ── Compute metrics (vs. original, since we have no GT in browser)
    const metrics = this.computeMetrics(data, imageData.data, W, H, degradationType, gate);

    return { canvas: upscaled, metrics, degradationType };
  },

  computeMetrics(enhanced, original, W, H, degradationType, gate) {
    // PSNR: using enhanced vs a scaled-back version of original
    // (Full GT not available in browser; compute self-reference quality)
    let mse = 0;
    const n = original.length / 4;
    for (let i = 0; i < original.length; i += 4) {
      for (let c = 0; c < 3; c++) {
        const d = (enhanced[i+c] || 0) - original[i+c];
        mse += d * d;
      }
    }
    mse /= (n * 3);
    const psnr = mse > 0 ? (10 * Math.log10(255*255 / mse)) : 99;

    // SSIM approximation (simplified)
    const lum1 = Gatekeeper.getLuminance(original);
    const lum2 = Gatekeeper.getLuminance(new Uint8ClampedArray(
      Array.from({length: enhanced.length}, (_, i) => {
        const ch = i % 4;
        return ch < 3 ? (enhanced[Math.floor(i/4)*4+ch] || 0) : 255;
      })
    ));

    let mu1 = 0, mu2 = 0;
    for (let i = 0; i < lum1.length; i++) { mu1 += lum1[i]; mu2 += lum2[i]; }
    mu1 /= lum1.length; mu2 /= lum2.length;

    let sig1 = 0, sig2 = 0, sig12 = 0;
    for (let i = 0; i < lum1.length; i++) {
      sig1  += (lum1[i]-mu1)**2;
      sig2  += (lum2[i]-mu2)**2;
      sig12 += (lum1[i]-mu1)*(lum2[i]-mu2);
    }
    sig1  = Math.sqrt(sig1  / lum1.length);
    sig2  = Math.sqrt(sig2  / lum2.length);
    sig12 = sig12 / lum1.length;

    const C1 = (0.01*255)**2, C2 = (0.03*255)**2;
    const ssim = (2*mu1*mu2+C1)*(2*sig12+C2) / ((mu1**2+mu2**2+C1)*(sig1**2+sig2**2+C2));

    // Fidelity score (1 - normalised mean abs diff)
    let mad = 0;
    for (let i = 0; i < original.length; i += 4) {
      for (let c = 0; c < 3; c++) mad += Math.abs((enhanced[i+c]||0) - original[i+c]);
    }
    mad /= (n * 3);
    const fidelity = Math.max(0, 1 - mad/128);

    return {
      psnr:        Math.round(psnr * 100) / 100,
      ssim:        Math.round(Math.abs(ssim) * 10000) / 10000,
      fidelity:    Math.round(fidelity * 10000) / 10000,
      blurScore:   gate.blurScore,
      brightness:  gate.brightness,
      fogScore:    gate.fogScore,
    };
  },
};

/** Yield to browser to allow UI repaints */
function tick() {
  return new Promise(resolve => setTimeout(resolve, 0));
}

// Export to window for use in app.js
window.Pipeline = Pipeline;
window.Gatekeeper = Gatekeeper;
window.DegradationClassifier = DegradationClassifier;
