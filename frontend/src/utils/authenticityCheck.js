// On-device photo authenticity check: real camera photo vs AI-generated,
// photo of a screen, or downloaded/screenshot. Mirrors
// backend/services/image_authenticity.py; the server re-checks every
// submission, this only warns the citizen early.
import * as ort from 'onnxruntime-web/webgpu';
import exifr from 'exifr';

ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions.web}/dist/`;

const MODEL_URL = '/models/authenticity.onnx';
const MODEL_CONFIG_URL = '/models/authenticity.json';
const CLASSES = ['real', 'ai_generated', 'screen_photo'];
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

const AI_MARKERS = [
  ['trainedAlgorithmicMedia', 'Metadata declares the image AI-generated'],
  ['c2pa', 'Content credentials from an AI tool'],
  ['Midjourney', 'Midjourney metadata'],
  ['DALL', 'DALL-E metadata'],
  ['Stable Diffusion', 'Stable Diffusion metadata'],
  ['ComfyUI', 'ComfyUI metadata'],
  ['Adobe Firefly', 'Adobe Firefly metadata'],
  ['Steps: ', 'AI generator settings embedded in the image'],
];

let classifierPromise = null;

async function createClassifier() {
  const res = await fetch(MODEL_CONFIG_URL);
  // Model not deployed yet: metadata checks only
  if (!res.ok || !(res.headers.get('content-type') || '').includes('json')) return null;
  const config = await res.json();
  const session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ['wasm'] });
  return { session, config };
}

function loadClassifier() {
  if (!classifierPromise) {
    classifierPromise = createClassifier().catch((err) => {
      console.warn('Authenticity model unavailable:', err);
      return null;
    });
  }
  return classifierPromise;
}

async function scanMetadata(file) {
  const head = new TextDecoder('latin1').decode(await file.slice(0, 2_000_000).arrayBuffer());
  const aiMarkers = AI_MARKERS.filter(([m]) => head.includes(m)).map(([, reason]) => reason);
  let exif = null;
  try {
    exif = await exifr.parse(file, { tiff: true, exif: true, xmp: false, iptc: false });
  } catch {
    exif = null;
  }
  return {
    aiMarkers,
    screenshot: /screenshot/i.test(head.slice(0, 200_000)),
    camera: exif && (exif.Make || exif.Model) ? `${exif.Make || ''} ${exif.Model || ''}`.trim() : null,
  };
}

/** Center + 4 corner crops after resizing the shorter side (matches training/serving). */
function cropsTensor(bitmap, crop, shortSide) {
  const scale = Math.min(1, shortSide / Math.min(bitmap.width, bitmap.height));
  let w = Math.round(bitmap.width * scale);
  let h = Math.round(bitmap.height * scale);
  w = Math.max(w, crop);
  h = Math.max(h, crop);
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bitmap, 0, 0, w, h);
  const origins = [
    [(w - crop) >> 1, (h - crop) >> 1], [0, 0], [w - crop, 0], [0, h - crop], [w - crop, h - crop],
  ];
  const plane = crop * crop;
  const data = new Float32Array(origins.length * 3 * plane);
  origins.forEach(([x, y], n) => {
    const px = ctx.getImageData(x, y, crop, crop).data;
    for (let i = 0; i < plane; i++) {
      for (let c = 0; c < 3; c++) {
        data[n * 3 * plane + c * plane + i] = (px[i * 4 + c] / 255 - MEAN[c]) / STD[c];
      }
    }
  });
  return new ort.Tensor('float32', data, [origins.length, 3, crop, crop]);
}

async function classify(file) {
  const classifier = await loadClassifier();
  if (!classifier) return null;
  const { session, config } = classifier;
  const bitmap = await createImageBitmap(file);
  const input = cropsTensor(bitmap, config.crop, config.shortSide);
  bitmap.close?.();
  const out = await session.run({ [session.inputNames[0]]: input });
  const logits = out[session.outputNames[0]].data;
  const n = logits.length / CLASSES.length;
  const probs = new Array(CLASSES.length).fill(0);
  for (let k = 0; k < n; k++) {
    const row = Array.from(logits.slice(k * CLASSES.length, (k + 1) * CLASSES.length));
    const max = Math.max(...row);
    const exps = row.map((v) => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    exps.forEach((e, i) => { probs[i] += e / sum / n; });
  }
  return { probs: Object.fromEntries(CLASSES.map((c, i) => [c, probs[i]])), threshold: config.threshold };
}

/**
 * Check one photo. Returns { verdict, flagged, reasons }.
 * flagged = strong evidence (AI markers, classifier, screenshot); a missing
 * camera EXIF alone is only a soft "downloaded" hint (forwarded photos lose it too).
 */
export async function checkPhoto(file, { live = false } = {}) {
  const meta = await scanMetadata(file);
  let model = null;
  try {
    model = await classify(file);
  } catch (err) {
    console.warn('Authenticity classification failed:', err);
  }
  const p = model?.probs;
  const t = model?.threshold ?? 0.5;
  if (meta.aiMarkers.length) return { verdict: 'ai_generated', flagged: true, reasons: meta.aiMarkers };
  if (p && p.ai_generated >= t) {
    return { verdict: 'ai_generated', flagged: true, reasons: [`Looks AI-generated (${Math.round(p.ai_generated * 100)}%)`] };
  }
  if (p && p.screen_photo >= t) {
    return { verdict: 'screen_photo', flagged: true, reasons: [`Looks like a photo of a screen (${Math.round(p.screen_photo * 100)}%)`] };
  }
  if (meta.screenshot) return { verdict: 'downloaded', flagged: true, reasons: ['This is a screenshot'] };
  if (!live && !meta.camera) {
    return { verdict: 'downloaded', flagged: false, reasons: ['No camera information in this photo'] };
  }
  return { verdict: 'real', flagged: false, reasons: [] };
}
