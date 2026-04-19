// ── Bootstrap ─────────────────────────────────────────────────────────────
(async function () {
  const params = new URLSearchParams(window.location.search);
  const networkId = params.get('id');
  if (!networkId) return;  // landing page — nothing to do

  // Fetch network info
  const registry = await (await fetch('networks/registry.json')).json();
  const entry = registry.find(e => e.id === networkId);
  if (!entry) {
    document.getElementById('net-name').textContent = 'Network not found';
    return;
  }
  const info = await (await fetch(entry.path)).json();

  // Page title
  document.title = info.name;

  // Header meta
  document.getElementById('net-name').textContent = info.name;
  document.getElementById('net-arch').textContent = info.architecture;
  document.getElementById('net-results').textContent = info.results.summary || '';

  // Load weights + set up try-it widget.
  // Two backends:
  //   - 'mnist'      → hand-rolled forward pass over JSON weight matrices (Network 1)
  //   - 'mnist-cnn'  → TensorFlow.js LayersModel loaded from tfjs_model/model.json
  // Wrapped in try/catch so a broken try-it widget doesn't prevent notes and the
  // notebook below from rendering.
  try {
    if (info.input_type === 'mnist') {
      const weights = await (await fetch(info.weights)).json();
      initTryIt(info, { kind: 'dense', weights });
    } else if (info.input_type === 'mnist-cnn') {
      if (typeof tf === 'undefined') {
        throw new Error('TensorFlow.js not loaded — check the CDN script tag');
      }
      const tfjsModel = await tf.loadLayersModel(info.tfjs_model);
      initTryIt(info, { kind: 'cnn', model: tfjsModel });
    } else if (info.input_type === 'coco-yolo') {
      let session = null;
      if (info.onnx_model) {
        if (typeof ort === 'undefined') throw new Error('onnxruntime-web not loaded');
        session = await ort.InferenceSession.create(info.onnx_model);
      }
      initYoloTryIt(info, session);
    }
  } catch (e) {
    console.error('Failed to set up try-it widget:', e);
  }

  // Load notes
  if (info.notes) {
    try {
      const md = await (await fetch(info.notes)).text();
      document.getElementById('notes-content').innerHTML = marked.parse(md);
      document.getElementById('notes-section').style.display = '';
    } catch (_) {}
  }

  // Load notebook
  if (info.notebook) {
    try {
      const nb = await (await fetch(info.notebook)).json();
      renderNotebook(nb, info.name);
      document.getElementById('nb-container').style.display = '';
    } catch (e) {
      console.error('Failed to load notebook:', e);
    }
  }
})();

// ── Try-it: Canvas + Prediction ───────────────────────────────────────────
// `backend` is one of:
//   { kind: 'dense', weights }  — hand-rolled MLP forward pass
//   { kind: 'cnn',   model   }  — TensorFlow.js LayersModel
function initTryIt(info, backend) {
  const section = document.getElementById('try-it-section');
  section.style.display = '';

  const canvas = document.getElementById('draw-canvas');
  const ctx = canvas.getContext('2d');
  const clearBtn = document.getElementById('clear-btn');
  const guessBtn = document.getElementById('guess-btn');
  const predDisplay = document.getElementById('prediction-display');
  const barChart = document.getElementById('bar-chart');

  // Canvas setup
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 18;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = '#fff';

  let isDrawing = false, lastX = 0, lastY = 0;

  function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    const src = e.touches ? e.touches[0] : e;
    return {
      x: (src.clientX - rect.left) * (canvas.width / rect.width),
      y: (src.clientY - rect.top) * (canvas.height / rect.height)
    };
  }

  canvas.addEventListener('mousedown', e => { isDrawing = true; const p = getPos(e); lastX = p.x; lastY = p.y; });
  canvas.addEventListener('mousemove', e => { if (!isDrawing) return; const p = getPos(e); ctx.beginPath(); ctx.moveTo(lastX, lastY); ctx.lineTo(p.x, p.y); ctx.stroke(); lastX = p.x; lastY = p.y; });
  canvas.addEventListener('mouseup', () => { isDrawing = false; ctx.beginPath(); });
  canvas.addEventListener('mouseleave', () => { isDrawing = false; ctx.beginPath(); });
  canvas.addEventListener('touchstart', e => { e.preventDefault(); isDrawing = true; const p = getPos(e); lastX = p.x; lastY = p.y; });
  canvas.addEventListener('touchmove', e => { e.preventDefault(); if (!isDrawing) return; const p = getPos(e); ctx.beginPath(); ctx.moveTo(lastX, lastY); ctx.lineTo(p.x, p.y); ctx.stroke(); lastX = p.x; lastY = p.y; });
  canvas.addEventListener('touchend', () => { isDrawing = false; ctx.beginPath(); });

  clearBtn.addEventListener('click', () => {
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    predDisplay.innerHTML = '<span class="prediction-label">Draw something and press Guess</span>';
    barChart.innerHTML = '';
  });

  guessBtn.disabled = false;
  guessBtn.addEventListener('click', () => {
    const size = (info.input_shape && info.input_shape[0]) || 28;
    const pixels = getPixels(canvas, size);
    let probs;
    if (backend.kind === 'dense') {
      probs = forwardPass(pixels, backend.weights, info.layers);
    } else {
      const input = tf.tensor4d(pixels, [1, size, size, 1]);
      const out = backend.model.predict(input);
      probs = Array.from(out.dataSync());
      input.dispose();
      out.dispose();
    }
    const pred = probs.indexOf(Math.max(...probs));

    predDisplay.innerHTML =
      `<div class="prediction-number">${pred}</div>` +
      `<div class="prediction-conf">${(probs[pred] * 100).toFixed(1)}% confidence</div>`;

    barChart.innerHTML = '';
    const top = Math.max(...probs);
    probs.forEach((p, d) => {
      const row = document.createElement('div');
      row.className = 'bar-row';
      row.innerHTML =
        `<span class="bar-digit">${d}</span>` +
        `<div class="bar-track"><div class="bar-fill ${d === pred ? 'top' : ''}" style="width:${(p / top * 100).toFixed(1)}%"></div></div>` +
        `<span class="bar-pct">${(p * 100).toFixed(1)}%</span>`;
      barChart.appendChild(row);
    });
  });
}

// ── Generic forward pass ──────────────────────────────────────────────────
const activations = {
  tanh: v => v.map(x => Math.tanh(x)),
  sigmoid: v => v.map(x => 1 / (1 + Math.exp(-x))),
  relu: v => v.map(x => Math.max(0, x)),
  softmax: v => {
    const mx = Math.max(...v);
    const e = v.map(x => Math.exp(x - mx));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map(x => x / s);
  },
  linear: v => v
};

function dotVecMat(vec, mat) {
  const m = mat[0].length;
  const out = new Float64Array(m);
  for (let j = 0; j < m; j++) {
    let s = 0;
    for (let i = 0; i < vec.length; i++) s += vec[i] * mat[i][j];
    out[j] = s;
  }
  return Array.from(out);
}

function addBias(vec, bias) {
  return vec.map((v, i) => v + bias[i]);
}

function forwardPass(input, weights, layers) {
  let current = Array.from(input);
  for (const layer of layers) {
    current = dotVecMat(current, weights[layer.weights]);
    if (layer.bias && weights[layer.bias]) {
      current = addBias(current, weights[layer.bias]);
    }
    const act = activations[layer.activation] || activations.linear;
    current = act(current);
  }
  return current;
}

// ── Get pixels from canvas at the network's expected resolution ───────────
function getPixels(canvas, size) {
  const n = size || 28;
  const s = document.createElement('canvas');
  s.width = s.height = n;
  s.getContext('2d').drawImage(canvas, 0, 0, n, n);
  const d = s.getContext('2d').getImageData(0, 0, n, n).data;
  const total = n * n;
  const p = new Float32Array(total);
  for (let i = 0; i < total; i++) p[i] = d[i * 4] / 255;
  return p;
}

// ── YOLO drag-and-drop try-it ─────────────────────────────────────────────
const YOLO_GRID        = 13;
const YOLO_CLASSES     = 80;
const YOLO_ANCHORS     = [[0.28, 0.22], [0.38, 0.48]];
const YOLO_CONF_THRESH = 0.5;
const YOLO_NMS_THRESH  = 0.4;

const COCO_CLASSES = [
  'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
  'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
  'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
  'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
  'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
  'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
  'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
  'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
  'remote','keyboard','cell phone','microwave','oven','toaster','sink',
  'refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
];

function yoloSigmoid(x) { return 1 / (1 + Math.exp(-x)); }

function yoloIOU(a, b) {
  const ax1 = a.cx - a.w / 2, ay1 = a.cy - a.h / 2;
  const ax2 = a.cx + a.w / 2, ay2 = a.cy + a.h / 2;
  const bx1 = b.cx - b.w / 2, by1 = b.cy - b.h / 2;
  const bx2 = b.cx + b.w / 2, by2 = b.cy + b.h / 2;
  const inter = Math.max(0, Math.min(ax2, bx2) - Math.max(ax1, bx1))
              * Math.max(0, Math.min(ay2, by2) - Math.max(ay1, by1));
  return inter / (a.w * a.h + b.w * b.h - inter + 1e-6);
}

function yoloNMS(boxes) {
  const sorted = [...boxes].sort((a, b) => b.conf - a.conf);
  const kept = [], skip = new Set();
  for (let i = 0; i < sorted.length; i++) {
    if (skip.has(i)) continue;
    kept.push(sorted[i]);
    for (let j = i + 1; j < sorted.length; j++) {
      if (!skip.has(j) && yoloIOU(sorted[i], sorted[j]) > YOLO_NMS_THRESH) skip.add(j);
    }
  }
  return kept;
}

async function runYoloModel(img, session) {
  // Resize to 416×416 and build NCHW Float32 tensor
  const s = document.createElement('canvas');
  s.width = s.height = 416;
  s.getContext('2d').drawImage(img, 0, 0, 416, 416);
  const px = s.getContext('2d').getImageData(0, 0, 416, 416).data;
  const input = new Float32Array(3 * 416 * 416);
  for (let i = 0; i < 416 * 416; i++) {
    input[i]               = px[i * 4]     / 255;
    input[416 * 416 + i]   = px[i * 4 + 1] / 255;
    input[2 * 416 * 416 + i] = px[i * 4 + 2] / 255;
  }

  const inputName = session.inputNames[0];
  const tensor = new ort.Tensor('float32', input, [1, 3, 416, 416]);
  const result = await session.run({ [inputName]: tensor });
  const out = Object.values(result)[0].data; // Float32, [1,170,13,13] flattened

  // out index for [channel c, row r, col c]: c*13*13 + r*13 + col
  const gs = YOLO_GRID * YOLO_GRID;
  const boxes = [];

  for (let a = 0; a < 2; a++) {
    const [aw, ah] = YOLO_ANCHORS[a];
    const off = a * 85;
    for (let row = 0; row < YOLO_GRID; row++) {
      for (let col = 0; col < YOLO_GRID; col++) {
        const ci = row * YOLO_GRID + col;
        const conf = yoloSigmoid(out[(off + 0) * gs + ci]);
        if (conf < YOLO_CONF_THRESH) continue;

        const px_ = yoloSigmoid(out[(off + 1) * gs + ci]);
        const py_ = yoloSigmoid(out[(off + 2) * gs + ci]);
        const pw  = out[(off + 3) * gs + ci];
        const ph  = out[(off + 4) * gs + ci];

        const cx = (col + px_) / YOLO_GRID;
        const cy = (row + py_) / YOLO_GRID;
        const w  = Math.exp(pw) * aw;
        const h  = Math.exp(ph) * ah;

        let bestCls = 0, bestScore = -Infinity;
        for (let c = 0; c < YOLO_CLASSES; c++) {
          const sc = yoloSigmoid(out[(off + 5 + c) * gs + ci]);
          if (sc > bestScore) { bestScore = sc; bestCls = c; }
        }

        boxes.push({ cx, cy, w, h, conf, cls: bestCls, clsName: COCO_CLASSES[bestCls], clsConf: bestScore });
      }
    }
  }

  return yoloNMS(boxes);
}

function yoloDrawResults(canvas, img, boxes) {
  canvas.width  = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);

  const colors = ['#ef4444','#3b82f6','#22c55e','#f59e0b','#a855f7','#ec4899','#14b8a6'];
  boxes.forEach((box, idx) => {
    const color = colors[box.cls % colors.length];
    const x = (box.cx - box.w / 2) * img.naturalWidth;
    const y = (box.cy - box.h / 2) * img.naturalHeight;
    const w = box.w * img.naturalWidth;
    const h = box.h * img.naturalHeight;

    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(2, img.naturalWidth / 300);
    ctx.strokeRect(x, y, w, h);

    const label = `${box.clsName} ${(box.conf * 100).toFixed(0)}%`;
    ctx.font = `bold ${Math.max(12, img.naturalWidth / 50)}px sans-serif`;
    const tw = ctx.measureText(label).width + 8;
    const th = Math.max(18, img.naturalWidth / 30);
    ctx.fillStyle = color;
    ctx.fillRect(x, Math.max(0, y - th), tw, th);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, x + 4, Math.max(th - 4, y - 4));
  });
}

function initYoloTryIt(info, session) {
  const section = document.getElementById('try-it-section');
  section.style.display = '';

  // Replace the MNIST draw canvas with the YOLO drop-zone UI
  section.querySelector('.try-it-layout').innerHTML = `
    <div class="yolo-ui">
      <label class="yolo-drop-zone" id="yolo-drop-zone">
        <div class="yolo-drop-icon">&#128247;</div>
        <div class="yolo-drop-text">Drop an image here or <span class="yolo-browse-link">browse</span></div>
        ${!session ? '<div class="yolo-no-model">Export model.onnx to enable inference</div>' : ''}
        <input type="file" id="yolo-file-input" accept="image/*" style="display:none">
      </label>
      <div id="yolo-status" class="yolo-status" style="display:none"></div>
      <div id="yolo-result-wrap" style="display:none">
        <canvas id="yolo-canvas" class="yolo-canvas"></canvas>
        <div id="yolo-detections" class="yolo-detections"></div>
      </div>
    </div>
  `;

  if (!session) return;

  const dropZone   = document.getElementById('yolo-drop-zone');
  const fileInput  = document.getElementById('yolo-file-input');
  const status     = document.getElementById('yolo-status');
  const resultWrap = document.getElementById('yolo-result-wrap');
  const canvas     = document.getElementById('yolo-canvas');
  const detections = document.getElementById('yolo-detections');

  async function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) return;
    status.textContent = 'Running model…';
    status.style.display = '';
    resultWrap.style.display = 'none';

    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = async () => {
      try {
        const boxes = await runYoloModel(img, session);
        yoloDrawResults(canvas, img, boxes);

        if (boxes.length === 0) {
          detections.innerHTML = '<span class="yolo-none">No detections above threshold — try a clearer photo</span>';
        } else {
          detections.innerHTML = boxes.map(b =>
            `<div class="yolo-chip">
               <span class="yolo-chip-name">${b.clsName}</span>
               <span class="yolo-chip-conf">${(b.conf * 100).toFixed(0)}%</span>
             </div>`
          ).join('');
        }

        status.style.display = 'none';
        resultWrap.style.display = '';
      } catch (err) {
        status.textContent = 'Error: ' + err.message;
        console.error(err);
      }
      URL.revokeObjectURL(url);
    };
    img.src = url;
  }

  dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', ()  => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    handleFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', () => handleFile(fileInput.files[0]));
}

// ── Pyodide kernel ────────────────────────────────────────────────────────
let pyodide = null;
let pyodideReady = false;
let pyodideBooting = false;
let runCounter = 0;

function setKernelStatus(status) {
  const dot = document.getElementById('kernel-dot');
  const label = document.getElementById('kernel-label');
  if (!dot || !label) return;
  dot.className = 'nb-kernel-dot ' + status;
  const labels = { idle: 'Python (idle)', busy: 'Python (busy)', error: 'Python (error)', loading: 'Python (loading…)' };
  label.textContent = labels[status] || status;
}

async function bootPyodide() {
  if (pyodideReady || pyodideBooting) return;
  pyodideBooting = true;
  setKernelStatus('loading');
  try {
    pyodide = await loadPyodide();
    await pyodide.loadPackage(['numpy', 'matplotlib']);
    await pyodide.runPythonAsync(`
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64

def _show_capture():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    print('__IMG__' + base64.b64encode(buf.read()).decode() + '__ENDIMG__')
    plt.close('all')
plt.show = _show_capture
`);
    pyodideReady = true;
    setKernelStatus('idle');
  } catch (e) {
    setKernelStatus('error');
    console.error('Pyodide failed:', e);
  }
  pyodideBooting = false;
}

async function runCellCode(code, outputContent, promptEl) {
  if (!pyodideReady) await bootPyodide();
  if (!pyodideReady) {
    outputContent.innerHTML = '<pre class="nb-error-text">Could not start Python runtime.</pre>';
    return;
  }

  runCounter++;
  const thisRun = runCounter;
  promptEl.textContent = 'In [*]:';
  outputContent.innerHTML = '<div class="nb-running-text">Running…</div>';
  setKernelStatus('busy');

  pyodide.runPython(`
import sys, io as _io
_buf = _io.StringIO()
sys.stdout = _buf
sys.stderr = _buf
`);

  let rawOutput = '';
  let hasError = false;

  try {
    await pyodide.loadPackagesFromImports(code);
    await pyodide.runPythonAsync(code);
  } catch (err) {
    hasError = true;
    rawOutput += err.message + '\n';
  }

  try {
    rawOutput = pyodide.runPython('_buf.getvalue()') + rawOutput;
    pyodide.runPython('sys.stdout = sys.__stdout__; sys.stderr = sys.__stderr__');
  } catch (_) {}

  promptEl.textContent = `In [${thisRun}]:`;
  renderCellOutput(rawOutput, outputContent, hasError);
  setKernelStatus('idle');
}

function renderCellOutput(text, container, isError) {
  container.innerHTML = '';
  if (!text.trim()) return;

  const parts = text.split(/(__IMG__[\s\S]*?__ENDIMG__)/);
  parts.forEach(part => {
    const imgMatch = part.match(/^__IMG__([\s\S]*?)__ENDIMG__$/);
    if (imgMatch) {
      const img = document.createElement('img');
      img.src = `data:image/png;base64,${imgMatch[1].trim()}`;
      img.className = 'nb-plot-img';
      container.appendChild(img);
    } else if (part.trim()) {
      const pre = document.createElement('pre');
      pre.className = isError ? 'nb-error-text' : 'nb-stdout';
      pre.textContent = part;
      container.appendChild(pre);
    }
  });
}

// ── Notebook renderer ─────────────────────────────────────────────────────
function autoResize(ta) {
  ta.style.height = 'auto';
  ta.style.height = ta.scrollHeight + 'px';
}

function renderNotebook(nb, title) {
  const container = document.getElementById('notebook-render');
  container.innerHTML = '';
  document.getElementById('nb-title').textContent = title || 'Notebook';

  (nb.cells || []).forEach(cell => {
    const src = Array.isArray(cell.source) ? cell.source.join('') : (cell.source || '');

    if (cell.cell_type === 'markdown') {
      const div = document.createElement('div');
      div.className = 'nb-md-cell';
      div.innerHTML = marked.parse(src);
      container.appendChild(div);

    } else if (cell.cell_type === 'code') {
      const wrapper = document.createElement('div');
      wrapper.className = 'nb-code-cell';

      // Input area
      const inputArea = document.createElement('div');
      inputArea.className = 'nb-input-area';

      const inPrompt = document.createElement('div');
      inPrompt.className = 'nb-in-prompt';
      inPrompt.textContent = 'In [ ]:';

      const textarea = document.createElement('textarea');
      textarea.className = 'nb-editor';
      textarea.value = src;
      textarea.spellcheck = false;
      textarea.addEventListener('input', () => autoResize(textarea));
      textarea.addEventListener('focus', () => wrapper.classList.add('selected'));
      textarea.addEventListener('blur', () => wrapper.classList.remove('selected'));
      textarea.addEventListener('keydown', e => {
        if (e.key === 'Enter' && e.shiftKey) { e.preventDefault(); runBtn.click(); return; }
        if (e.key === 'Tab') {
          e.preventDefault();
          const s = textarea.selectionStart, en = textarea.selectionEnd;
          textarea.value = textarea.value.slice(0, s) + '    ' + textarea.value.slice(en);
          textarea.selectionStart = textarea.selectionEnd = s + 4;
        }
      });

      const runBtn = document.createElement('button');
      runBtn.className = 'nb-run-btn';
      runBtn.innerHTML = '&#9654;';
      runBtn.title = 'Run cell (Shift+Enter)';

      inputArea.appendChild(inPrompt);
      inputArea.appendChild(textarea);
      inputArea.appendChild(runBtn);

      // Output area
      const outputArea = document.createElement('div');
      outputArea.className = 'nb-output-area';

      const outPrompt = document.createElement('div');
      outPrompt.className = 'nb-out-prompt';

      const outputContent = document.createElement('div');
      outputContent.className = 'nb-output-content';

      // Render saved outputs from the .ipynb
      (cell.outputs || []).forEach(out => {
        if (out.output_type === 'stream') {
          const text = Array.isArray(out.text) ? out.text.join('') : out.text;
          const pre = document.createElement('pre');
          pre.className = 'nb-stdout';
          pre.textContent = text;
          outputContent.appendChild(pre);
        } else if (out.output_type === 'display_data' || out.output_type === 'execute_result') {
          const data = out.data || {};
          if (data['image/png']) {
            const img = document.createElement('img');
            img.src = `data:image/png;base64,${data['image/png']}`;
            img.className = 'nb-plot-img';
            outputContent.appendChild(img);
          } else if (data['text/plain']) {
            const text = Array.isArray(data['text/plain']) ? data['text/plain'].join('') : data['text/plain'];
            const pre = document.createElement('pre');
            pre.className = 'nb-stdout';
            pre.textContent = text;
            outputContent.appendChild(pre);
          }
        }
      });

      outputArea.appendChild(outPrompt);
      outputArea.appendChild(outputContent);
      wrapper.appendChild(inputArea);
      wrapper.appendChild(outputArea);
      container.appendChild(wrapper);

      // Wire run button
      runBtn.addEventListener('click', async () => {
        wrapper.classList.add('running');
        outPrompt.textContent = '';
        await runCellCode(textarea.value, outputContent, inPrompt);
        wrapper.classList.remove('running');
      });

      requestAnimationFrame(() => autoResize(textarea));
    }
  });

  // Run all
  document.getElementById('run-all-btn').onclick = async () => {
    for (const cell of container.querySelectorAll('.nb-code-cell')) {
      const runBtn = cell.querySelector('.nb-run-btn');
      if (runBtn) runBtn.click();
      await new Promise(r => setTimeout(r, 50));
    }
  };

  // Restart kernel
  document.getElementById('restart-btn').onclick = async () => {
    pyodide = null;
    pyodideReady = false;
    pyodideBooting = false;
    setKernelStatus('loading');
    runCounter = 0;
    container.querySelectorAll('.nb-in-prompt').forEach(p => p.textContent = 'In [ ]:');
    container.querySelectorAll('.nb-output-content').forEach(el => el.innerHTML = '');
    await bootPyodide();
  };
}
