/**
 * ui.js
 * ClearSight v4 — UI Controller
 *
 * Manages: logging, pipeline strip animation, compare slider,
 * toggle checkboxes, loading overlay, metrics display.
 */

'use strict';

/* ═══════════════════════════════════════════════════════════
   LOGGER
   ═══════════════════════════════════════════════════════════ */
const Logger = {
  _el: null,

  init(elementId) {
    this._el = document.getElementById(elementId);
  },

  _ts() {
    const d = new Date();
    return [d.getHours(), d.getMinutes(), d.getSeconds()]
      .map(n => String(n).padStart(2, '0'))
      .join(':');
  },

  write(msg, type = '') {
    if (!this._el) return;
    const row = document.createElement('div');
    row.className = 'log-line';
    row.innerHTML = `<span class="log-ts">${this._ts()}</span><span class="log-msg ${type}">${msg}</span>`;
    this._el.appendChild(row);
    this._el.scrollTop = this._el.scrollHeight;
  },

  info   (msg) { this.write(msg, 'info');    },
  success(msg) { this.write(msg, 'success'); },
  warn   (msg) { this.write(msg, 'warn');    },
  accent (msg) { this.write(msg, 'accent');  },
  plain  (msg) { this.write(msg, '');        },

  divider() { this.write('─'.repeat(48), ''); },
};


/* ═══════════════════════════════════════════════════════════
   PIPELINE STRIP ANIMATOR
   ═══════════════════════════════════════════════════════════ */
const PipelineStrip = {
  _steps: [],

  init() {
    this._steps = Array.from(document.querySelectorAll('[data-step]'));
  },

  setActive(index) {
    this._steps.forEach((el, i) => {
      el.classList.remove('active', 'done');
      if (i < index)  el.classList.add('done');
      if (i === index) el.classList.add('active');
    });
  },

  setAllDone() {
    this._steps.forEach(el => {
      el.classList.remove('active');
      el.classList.add('done');
    });
  },

  reset() {
    this._steps.forEach(el => el.classList.remove('active', 'done'));
  },
};


/* ═══════════════════════════════════════════════════════════
   LOADING OVERLAY
   ═══════════════════════════════════════════════════════════ */
const Loader = {
  _overlay: null,
  _text:    null,
  _steps:   null,

  init() {
    this._overlay = document.getElementById('loadingOverlay');
    this._text    = document.getElementById('loadingText');
    this._steps   = document.getElementById('loadingSteps');
  },

  show(text = 'PROCESSING...', sub = '') {
    this._text.textContent  = text;
    this._steps.textContent = sub;
    this._overlay.classList.add('show');
  },

  update(text, sub = '') {
    this._text.textContent  = text;
    this._steps.textContent = sub;
  },

  hide() {
    this._overlay.classList.remove('show');
  },
};


/* ═══════════════════════════════════════════════════════════
   TOGGLE CHECKBOXES
   ═══════════════════════════════════════════════════════════ */
const Toggles = {
  _state: { fidelity: true, clahe: true, denoise: true },

  init() {
    this._bind('toggleFidelity', 'fidelityBox', 'fidelity');
    this._bind('toggleClahe',    'claheBox',    'clahe');
    this._bind('toggleDenoise',  'denoiseBox',  'denoise');
  },

  _bind(wrapperId, boxId, key) {
    const wrapper = document.getElementById(wrapperId);
    const box     = document.getElementById(boxId);
    if (!wrapper || !box) return;
    wrapper.addEventListener('click', () => {
      this._state[key] = !this._state[key];
      box.classList.toggle('checked', this._state[key]);
    });
  },

  get(key) { return this._state[key]; },
  getAll()  { return { ...this._state }; },
};


/* ═══════════════════════════════════════════════════════════
   COMPARE SLIDER
   ═══════════════════════════════════════════════════════════ */
const CompareSlider = {
  _container:  null,
  _enhanced:   null,
  _line:       null,
  _origImg:    null,
  _enhImg:     null,
  _dragging:   false,
  _pos:        50,

  init() {
    this._container = document.getElementById('compareContainer');
    this._enhanced  = document.getElementById('compareEnhanced');
    this._line      = document.getElementById('compareLine');
    this._origImg   = document.getElementById('compareOrigImg');
    this._enhImg    = document.getElementById('compareEnhImg');
    if (!this._container) return;
    this._bindEvents();
  },

  load(origSrc, enhSrc) {
    this._origImg.src = origSrc;
    this._enhImg.src  = enhSrc;
    this._setPos(50);
  },

  _setPos(pct) {
    this._pos = Math.max(2, Math.min(98, pct));

    // Clip overlay width
    this._enhanced.style.width = this._pos + '%';

    // ✅ FIX: Keep inner image pinned to full container width
    // so it doesn't shrink as the clip gets smaller
    const containerW = this._container.getBoundingClientRect().width;
    this._enhImg.style.width    = containerW + 'px';
    this._enhImg.style.position = 'absolute';
    this._enhImg.style.top      = '0';
    this._enhImg.style.left     = '0';

    // Move divider line
    this._line.style.left = this._pos + '%';
  },

  _getPos(e) {
    const rect    = this._container.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    return ((clientX - rect.left) / rect.width) * 100;
  },

  _bindEvents() {
    this._container.addEventListener('mousedown', e => {
      this._dragging = true;
      this._setPos(this._getPos(e));
    });
    document.addEventListener('mousemove', e => {
      if (this._dragging) this._setPos(this._getPos(e));
    });
    document.addEventListener('mouseup', () => { this._dragging = false; });

    this._container.addEventListener('touchstart', e => {
      this._dragging = true;
      this._setPos(this._getPos(e));
    }, { passive: true });
    document.addEventListener('touchmove', e => {
      if (this._dragging) this._setPos(this._getPos(e));
    }, { passive: true });
    document.addEventListener('touchend', () => { this._dragging = false; });

    // ✅ FIX: Re-apply on window resize so image width stays correct
    window.addEventListener('resize', () => {
      this._setPos(this._pos);
    });
  },
};

/* ═══════════════════════════════════════════════════════════
   METRICS DISPLAY
   ═══════════════════════════════════════════════════════════ */
const Metrics = {
  _bar: null,

  init() {
    this._bar = document.getElementById('metricsBar');
  },

  show(metrics, degradationType) {
    document.getElementById('m-psnr').textContent     = metrics.psnr + ' dB';
    document.getElementById('m-ssim').textContent     = metrics.ssim;
    document.getElementById('m-mode').textContent     = degradationType;
    document.getElementById('m-fidelity').textContent = metrics.fidelity;
    this._bar.style.display = 'grid';
  },

  hide() {
    this._bar.style.display = 'none';
  },
};


/* ═══════════════════════════════════════════════════════════
   OUTPUT SECTION
   ═══════════════════════════════════════════════════════════ */
const OutputSection = {
  _empty:   null,
  _results: null,

  init() {
    this._empty   = document.getElementById('emptyOutput');
    this._results = document.getElementById('resultsOutput');
  },

  showEmpty() {
    this._empty.style.display   = 'block';
    this._results.style.display = 'none';
  },

  showResults(enhancedDataUrl) {
    document.getElementById('outputImg').src  = enhancedDataUrl;
    document.getElementById('dlEnhanced').href = enhancedDataUrl;
    this._empty.style.display   = 'none';
    this._results.style.display = 'block';
  },
};


/* ═══════════════════════════════════════════════════════════
   DROP ZONE
   ═══════════════════════════════════════════════════════════ */
const DropZone = {
  _zone:    null,
  _input:   null,
  _preview: null,
  _img:     null,
  _onFile:  null,

  init(onFile) {
    this._zone    = document.getElementById('dropZone');
    this._input   = document.getElementById('fileInput');
    this._preview = document.getElementById('inputPreview');
    this._img     = document.getElementById('inputImg');
    this._onFile  = onFile;

    const browseBtn = document.getElementById('browseBtn');
    browseBtn.addEventListener('click', e => { e.stopPropagation(); this._input.click(); });

    this._input.addEventListener('change', e => this._handleFile(e.target.files[0]));
    this._zone.addEventListener('click',   () => this._input.click());

    this._zone.addEventListener('dragover',  e => { e.preventDefault(); this._zone.classList.add('dragover'); });
    this._zone.addEventListener('dragleave', ()=> { this._zone.classList.remove('dragover'); });
    this._zone.addEventListener('drop', e => {
      e.preventDefault();
      this._zone.classList.remove('dragover');
      this._handleFile(e.dataTransfer.files[0]);
    });
  },

  _handleFile(file) {
    if (!file || !file.type.startsWith('image/')) {
      Logger.warn('Invalid file type — please upload an image.');
      return;
    }
    const reader = new FileReader();
    reader.onload = e => {
      const dataUrl = e.target.result;
      this._img.src = dataUrl;
      this._zone.style.display    = 'none';
      this._preview.style.display = 'flex';
      document.getElementById('inputPanel').classList.add('lit');
      if (this._onFile) this._onFile(file, dataUrl, dataUrl.split(',')[1]);
    };
    reader.readAsDataURL(file);
  },

  reset() {
    this._zone.style.display    = 'flex';
    this._preview.style.display = 'none';
    this._input.value = '';
    document.getElementById('inputPanel').classList.remove('lit');
    this._img.src = '';
  },
};


// Export to window
window.UI = { Logger, PipelineStrip, Loader, Toggles, CompareSlider, Metrics, OutputSection, DropZone };
