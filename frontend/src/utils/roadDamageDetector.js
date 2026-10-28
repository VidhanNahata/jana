// On-device road-damage detection for dashcam mode (potholes, cracks, alligator
// cracking) with ONNX Runtime Web. Class names, per-class thresholds and the
// taxonomy subcategory of each class ship in road_damage.json next to the model.
import * as ort from 'onnxruntime-web/webgpu';

ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions.web}/dist/`;

const MODEL_URL = '/models/road_damage.onnx';
const CONFIG_URL = '/models/road_damage.json';
const NMS_IOU_THRESHOLD = 0.7;

let promise = null;

async function create() {
  const config = await fetch(CONFIG_URL).then((r) => {
    if (!r.ok) throw new Error('road-damage model not installed');
    return r.json();
  });
  const providers = 'gpu' in navigator ? ['webgpu', 'wasm'] : ['wasm'];
  let session;
  try {
    session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: providers });
  } catch (err) {
    if (providers[0] !== 'webgpu') throw err;
    session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ['wasm'] });
  }
  const size = config.inputSize;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  return { session, config, canvas, ctx, size };
}

export function loadRoadDamage() {
  if (!promise) {
    promise = create().catch((err) => {
      promise = null;
      throw err;
    });
  }
  return promise;
}

function iou(a, b) {
  const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const inter = ix * iy;
  return inter / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter + 1e-9);
}

/**
 * Detect road damage in a video frame.
 * Returns [{ x1, y1, x2, y2 (0-1), confidence, classIndex, name, label, subcategoryId }]
 */
export async function detectRoadDamage(source) {
  const { session, config, ctx, size } = await loadRoadDamage();
  const srcW = source.videoWidth || source.naturalWidth || source.width;
  const srcH = source.videoHeight || source.naturalHeight || source.height;
  const scale = Math.min(size / srcW, size / srcH);
  const newW = Math.round(srcW * scale);
  const newH = Math.round(srcH * scale);
  const padX = Math.round((size - newW) / 2 - 0.1);
  const padY = Math.round((size - newH) / 2 - 0.1);
  ctx.fillStyle = 'rgb(114,114,114)';
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(source, padX, padY, newW, newH);

  const { data } = ctx.getImageData(0, 0, size, size);
  const area = size * size;
  const input = new Float32Array(3 * area);
  for (let i = 0; i < area; i++) {
    input[i] = data[i * 4] / 255;
    input[i + area] = data[i * 4 + 1] / 255;
    input[i + 2 * area] = data[i * 4 + 2] / 255;
  }
  const results = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, size, size]) });
  // output0: [1, 4 box + one score per class, anchors]
  const output = results[session.outputNames[0]];
  const anchors = output.dims[2];
  const p = output.data;
  const classes = config.classes;
  const clamp = (v, max) => Math.min(Math.max(v, 0), max);

  const candidates = [];
  for (let i = 0; i < anchors; i++) {
    let best = 0;
    let score = p[4 * anchors + i];
    for (let k = 1; k < classes.length; k++) {
      const s = p[(4 + k) * anchors + i];
      if (s > score) { score = s; best = k; }
    }
    if (classes[best].enabled === false || score < classes[best].threshold) continue;
    const cx = p[i], cy = p[anchors + i], w = p[2 * anchors + i], h = p[3 * anchors + i];
    candidates.push({
      x1: clamp((cx - w / 2 - padX) / scale, srcW) / srcW,
      y1: clamp((cy - h / 2 - padY) / scale, srcH) / srcH,
      x2: clamp((cx + w / 2 - padX) / scale, srcW) / srcW,
      y2: clamp((cy + h / 2 - padY) / scale, srcH) / srcH,
      confidence: score,
      classIndex: best,
      name: classes[best].name,
      label: classes[best].label_en,
      subcategoryId: classes[best].subcategory_id
    });
  }
  Object.values(results).forEach((t) => t.dispose?.());
  candidates.sort((a, b) => b.confidence - a.confidence);
  const keep = [];
  for (const c of candidates) {
    if (!keep.some((k) => k.classIndex === c.classIndex && iou(k, c) > NMS_IOU_THRESHOLD)) keep.push(c);
  }
  return keep;
}
