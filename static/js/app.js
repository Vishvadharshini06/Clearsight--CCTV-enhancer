/**
 * app.js
 * ClearSight v4 — Application Entry Point
 *
 * Wires together UI controllers and the enhancement pipeline.
 * All event listeners are registered inside DOMContentLoaded.
 */

'use strict';

document.addEventListener('DOMContentLoaded', () => {

  /* ─────────────────────────────────────────────────────────
     DESTRUCTURE UI MODULES
     ───────────────────────────────────────────────────────── */
  const { Logger, PipelineStrip, Loader, Toggles, CompareSlider, Metrics, OutputSection, DropZone } = window.UI;

  /* ─────────────────────────────────────────────────────────
     STATE
     ───────────────────────────────────────────────────────── */
  let currentFile      = null;   // File object
  let currentDataUrl   = null;   // Original image data URL
  let enhancedDataUrl  = null;   // Enhanced image data URL
  let currentMetrics   = null;
  let currentDegType   = null;
  let isProcessing     = false;

  /* ─────────────────────────────────────────────────────────
     INITIALISE UI MODULES
     ───────────────────────────────────────────────────────── */
  Logger.init('analysisLog');
  PipelineStrip.init();
  Loader.init();
  Toggles.init();
  CompareSlider.init();
  Metrics.init();
  OutputSection.init();

  DropZone.init((file, dataUrl, base64) => {
    currentFile    = file;
    currentDataUrl = dataUrl;
    enhancedDataUrl = null;

    document.getElementById('enhanceBtn').disabled = false;

    Logger.divider();
    Logger.info(`Frame loaded: ${file.name}`);
    Logger.info(`Size: ${(file.size / 1024).toFixed(1)} KB  |  Type: ${file.type}`);
    Logger.accent('Ready — click INITIATE ENHANCEMENT PIPELINE');
  });

  /* ─────────────────────────────────────────────────────────
     PIPELINE STEP DEFINITIONS
     ───────────────────────────────────────────────────────── */
  const STEPS = [
    { label: 'PHASE 1 — GATEKEEPER ANALYSIS',         log: 'Running Gatekeeper: blur_score · brightness · fog_score...' },
    { label: 'PHASE 2 — DEGRADATION CLASSIFICATION',  log: 'Classifying degradation type (FOG/NIGHT/BLUR/RAIN/CLEAR)...' },
    { label: 'PHASE 3 — TEMPORAL FUSION SR',          log: 'Applying temporal fusion + CLAHE (clipLimit=3.0, tile=8×8)...' },
    { label: 'PHASE 4 — DOMAIN ENHANCEMENT',          log: 'Running domain-specific enhancement module...' },
    { label: 'PHASE 5 — ESRGAN SUPER-RESOLUTION',     log: 'RealESRGAN upscaling with post-sharpen passes...' },
    { label: 'PHASE 6 — FIDELITY GUARD',              log: 'White-balance · contrast-stretch · micro-detail recovery...' },
    { label: 'PHASE 7 — EVIDENCE LOGGING',            log: 'Hashing output · writing metadata JSON...' },
  ];

  /* ─────────────────────────────────────────────────────────
     ENHANCE BUTTON
     ───────────────────────────────────────────────────────── */
  const enhanceBtn = document.getElementById('enhanceBtn');

  enhanceBtn.addEventListener('click', async () => {
    if (!currentDataUrl || isProcessing) return;

    isProcessing = true;
    enhanceBtn.disabled = true;

    // Collect options
    const opts = {
      mode:          document.getElementById('modeSelect').value,
      scale:         parseInt(document.getElementById('scaleSelect').value) || 4,
      depth:         document.getElementById('depthSelect').value,
      quality:       document.getElementById('qualitySelect').value,
      clahe:         Toggles.get('clahe'),
      denoise:       Toggles.get('denoise'),
      fidelityGuard: Toggles.get('fidelity'),
    };

    Logger.divider();
    Logger.accent('PIPELINE INITIATED — ClearSight v4');
    Logger.info(`Mode: ${opts.mode.toUpperCase()}  |  Scale: ${opts.scale}×  |  Quality: ${opts.quality.toUpperCase()}`);
    Logger.info(`CLAHE: ${opts.clahe}  |  Denoise: ${opts.denoise}  |  FidelityGuard: ${opts.fidelityGuard}`);

    Loader.show('INITIALIZING PIPELINE...', 'Step 0 of ' + STEPS.length);
    PipelineStrip.reset();

    // Load image element
    const img = new Image();
    img.src = currentDataUrl;
    await new Promise(resolve => { img.onload = resolve; });

    // Run pipeline
    try {
      const result = await Pipeline.run(img, {
        ...opts,
        onStep: (stepIdx) => {
          PipelineStrip.setActive(stepIdx);
          const s = STEPS[stepIdx];
          if (s) {
            Loader.update(s.label, `Step ${stepIdx + 1} of ${STEPS.length}`);
            Logger.info(s.log);
          }
        },
      });

      // ── Success
      PipelineStrip.setAllDone();

      // Convert canvas to data URL
      enhancedDataUrl = result.canvas.toDataURL('image/png', 1.0);
      currentMetrics  = result.metrics;
      currentDegType  = result.degradationType;

      // Show evidence step log
      Logger.divider();
      Logger.success(`PIPELINE COMPLETE ✓`);
      Logger.success(`Degradation: ${result.degradationType}  |  Scale: ${opts.scale}×`);
      Logger.success(`PSNR: ${result.metrics.psnr} dB  |  SSIM: ${result.metrics.ssim}  |  Fidelity: ${result.metrics.fidelity}`);
      Logger.info(`Output: ${result.canvas.width}×${result.canvas.height}px`);
      Logger.accent(`Evidence: case_${Date.now()}`);

      // Update UI
      Metrics.show(result.metrics, result.degradationType);
      OutputSection.showResults(enhancedDataUrl);
      CompareSlider.load(currentDataUrl, enhancedDataUrl);

    } catch (err) {
      Logger.warn('Pipeline error: ' + err.message);
      console.error(err);
    }

    Loader.hide();
    isProcessing = false;
    enhanceBtn.disabled = false;
  });

  /* ─────────────────────────────────────────────────────────
     EXPORT EVIDENCE REPORT
     ───────────────────────────────────────────────────────── */
  document.getElementById('dlReport').addEventListener('click', () => {
    if (!currentMetrics) return;

    const caseId = `case_${Date.now()}`;
    const report = {
      system:    'ClearSight v4',
      version:   '4.0.0',
      case_id:   caseId,
      timestamp: new Date().toISOString(),
      pipeline: [
        'Gatekeeper',
        'DegradationClassifier',
        'TemporalFusionSR',
        'DomainEnhancement',
        'ESRGANEnhancer',
        'FidelityGuard',
        'EvidenceLogger',
      ],
      settings: {
        mode:          document.getElementById('modeSelect').value,
        scale:         document.getElementById('scaleSelect').value,
        depth:         document.getElementById('depthSelect').value,
        quality:       document.getElementById('qualitySelect').value,
        clahe:         Toggles.get('clahe'),
        denoise:       Toggles.get('denoise'),
        fidelity_guard: Toggles.get('fidelity'),
      },
      degradation_type: currentDegType,
      metrics: {
        psnr_db:       currentMetrics.psnr,
        ssim:          currentMetrics.ssim,
        fidelity_score: currentMetrics.fidelity,
        blur_score:    currentMetrics.blurScore,
        brightness:    currentMetrics.brightness,
        fog_score:     currentMetrics.fogScore,
      },
      source_file: currentFile ? {
        name: currentFile.name,
        size_kb: Math.round(currentFile.size / 1024),
        type: currentFile.type,
      } : null,
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url  = URL.createObjectURL(blob);
    const a    = Object.assign(document.createElement('a'), {
      href:     url,
      download: `clearsight_evidence_${caseId}.json`,
    });
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    Logger.success(`Evidence report exported: ${caseId}.json`);
  });

  /* ─────────────────────────────────────────────────────────
     RESET
     ───────────────────────────────────────────────────────── */
  document.getElementById('resetBtn').addEventListener('click', () => {
    currentFile     = null;
    currentDataUrl  = null;
    enhancedDataUrl = null;
    currentMetrics  = null;
    currentDegType  = null;

    DropZone.reset();
    PipelineStrip.reset();
    Metrics.hide();
    OutputSection.showEmpty();
    enhanceBtn.disabled = true;

    Logger.divider();
    Logger.accent('System reset — ready for new frame.');
  });

  /* ─────────────────────────────────────────────────────────
     INITIAL LOG
     ───────────────────────────────────────────────────────── */
  // Clear default log content and write fresh entries
  document.getElementById('analysisLog').innerHTML = '';
  Logger.plain('ClearSight v4 initialized — all modules ready.');
  Logger.plain('Pipeline: Gatekeeper · DehazeKernel · ESRGANEnhancer · FidelityGuard');
  Logger.plain('Datasets: LOL (Night) · RESIDE (Fog) · REDS (Blur+SR) · CCPD (Plate)');
  Logger.accent('Drop a CCTV frame to begin.');

});
