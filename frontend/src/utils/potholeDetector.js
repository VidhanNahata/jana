// On-device pothole detection with ONNX Runtime Web.
// Runs the YOLO model in the browser (WebGPU when available, else WebAssembly)
// so live scanning doesn't depend on a slow server round trip per frame.
import * as ort from 'onnxruntime-web/webgpu';

// WASM binaries are fetched from the CDN build matching the installed package
ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions.web}/dist/`;

const MODEL_URL = '/models/pothole.onnx';
// Threshold + input size ship next to the model so a retrained model can change them
const MODEL_CONFIG_URL = '/models/pothole.json';
const NMS_IOU_THRESHOLD = 0.7;

let detectorPromise = null;

async function createDetector() {
  const config = await fetch(MODEL_CONFIG_URL).then((r) => r.json());
  const providers = 'gpu' in navigator ? ['webgpu', 'wasm'] : ['wasm'];
  let session;
  try {
    session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: providers });
  } catch (err) {
    if (providers[0] !== 'webgpu') throw err;
    // Some WebGPU implementations reject ops the model needs; WASM always works
    session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ['wasm'] });
  }

  const size = config.inputSize;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });

  return { session, config, canvas, ctx, size };
}

/** Load (once) and return the detector; safe to call repeatedly. */
export function loadDetector() {
  if (!detectorPromise) {
    detectorPromise = createDetector().catch((err) => {
      detectorPromise = null; // allow a retry after e.g. a network blip
      throw err;
    });
  }
  return detectorPromise;
}

function nms(boxes, iouThreshold) {
  const sorted = [...boxes].sort((a, b) => b.confidence - a.confidence);
  const keep = [];
  for (const box of sorted) {
    const overlaps = keep.some((k) => {
      const ix = Math.max(0, Math.min(box.x2, k.x2) - Math.max(box.x1, k.x1));
      const iy = Math.max(0, Math.min(box.y2, k.y2) - Math.max(box.y1, k.y1));
      const inter = ix * iy;
      const union = (box.x2 - box.x1) * (box.y2 - box.y1) + (k.x2 - k.x1) * (k.y2 - k.y1) - inter;
      return inter / (union + 1e-9) > iouThreshold;
    });
    if (!overlaps) keep.push(box);
  }
  return keep;
}

// Mask resolution inside each box, in proto pixels (proto is 1/4 of the input size)
const MAX_MASK_SIDE = 160;

/**
 * Pothole mask inside a box: sigmoid(coeffs . proto) > 0.5, sampled over the
 * box only. Returns { width, height, data: Uint8Array (1 = pothole) } covering
 * exactly the box, so it can be drawn stretched onto the box rectangle.
 */
function decodeMask(box, proto, { scaleX, scaleY, padX, padY, size }) {
  const [, numCoeffs, protoH, protoW] = proto.dims;
  const p = proto.data;
  const plane = protoH * protoW;
  // Box corners in proto pixels
  const px1 = ((box.x1 * scaleX + padX) / size) * protoW;
  const py1 = ((box.y1 * scaleY + padY) / size) * protoH;
  const px2 = ((box.x2 * scaleX + padX) / size) * protoW;
  const py2 = ((box.y2 * scaleY + padY) / size) * protoH;
  const width = Math.max(1, Math.min(MAX_MASK_SIDE, Math.round(px2 - px1)));
  const height = Math.max(1, Math.min(MAX_MASK_SIDE, Math.round(py2 - py1)));
  const data = new Uint8Array(width * height);
  for (let j = 0; j < height; j++) {
    const y = Math.min(protoH - 1, Math.floor(py1 + ((j + 0.5) * (py2 - py1)) / height));
    for (let i = 0; i < width; i++) {
      const x = Math.min(protoW - 1, Math.floor(px1 + ((i + 0.5) * (px2 - px1)) / width));
      let v = 0;
      for (let k = 0; k < numCoeffs; k++) v += box.coeffs[k] * p[k * plane + y * protoW + x];
      data[j * width + i] = v > 0 ? 1 : 0; // sigmoid(v) > 0.5
    }
  }
  return { width, height, data };
}

/**
 * Detect potholes in a video frame (or image/canvas).
 * Returns the same shape as the backend /api/detect/pothole endpoint:
 * { detected, count, max_confidence, boxes: [{x1,y1,x2,y2,confidence}] (0-1), threshold }
 * With a segmentation model each box also has `mask` (see decodeMask).
 */
export async function detectPotholes(source) {
  const { session, config, canvas, ctx, size } = await loadDetector();
  const srcW = source.videoWidth || source.naturalWidth || source.width;
  const srcH = source.videoHeight || source.naturalHeight || source.height;

  // Letterbox: keep aspect ratio, pad with grey (114) like Ultralytics
  const scale = Math.min(size / srcW, size / srcH);
  const newW = Math.round(srcW * scale);
  const newH = Math.round(srcH * scale);
  const padX = Math.round((size - newW) / 2 - 0.1);
  const padY = Math.round((size - newH) / 2 - 0.1);
  ctx.fillStyle = 'rgb(114,114,114)';
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(source, padX, padY, newW, newH);

  // HWC RGBA bytes -> CHW float32 RGB in [0,1]
  const { data } = ctx.getImageData(0, 0, size, size);
  const area = size * size;
  const input = new Float32Array(3 * area);
  for (let i = 0; i < area; i++) {
    input[i] = data[i * 4] / 255;
    input[i + area] = data[i * 4 + 1] / 255;
    input[i + 2 * area] = data[i * 4 + 2] / 255;
  }

  const feeds = { [session.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, size, size]) };
  const results = await session.run(feeds);
  // output0: [1, 4 box + 1 class (+ 32 mask coeffs for seg models), anchors]
  const output = results[session.outputNames[0]];
  const [, channels, anchors] = output.dims;
  const preds = output.data;
  const threshold = config.threshold;
  // Segmentation models have a second output: mask prototypes [1, 32, size/4, size/4]
  const proto = session.outputNames.length > 1 ? results[session.outputNames[1]] : null;
  const numCoeffs = proto ? channels - 5 : 0;

  const candidates = [];
  for (let i = 0; i < anchors; i++) {
    const score = preds[4 * anchors + i];
    if (score < threshold) continue;
    const cx = preds[i];
    const cy = preds[anchors + i];
    const w = preds[2 * anchors + i];
    const h = preds[3 * anchors + i];
    const clamp = (v, max) => Math.min(Math.max(v, 0), max);
    const box = {
      x1: clamp((cx - w / 2 - padX) / scale, srcW) / srcW,
      y1: clamp((cy - h / 2 - padY) / scale, srcH) / srcH,
      x2: clamp((cx + w / 2 - padX) / scale, srcW) / srcW,
      y2: clamp((cy + h / 2 - padY) / scale, srcH) / srcH,
      confidence: score
    };
    if (numCoeffs) {
      box.coeffs = Array.from({ length: numCoeffs }, (_, k) => preds[(5 + k) * anchors + i]);
    }
    candidates.push(box);
  }

  const boxes = nms(candidates, NMS_IOU_THRESHOLD);
  if (proto) {
    const toLetterbox = { scaleX: srcW * scale, scaleY: srcH * scale, padX, padY, size };
    for (const box of boxes) {
      box.mask = decodeMask(box, proto, toLetterbox);
      delete box.coeffs;
    }
  }
  Object.values(results).forEach((t) => t.dispose?.());
  return {
    detected: boxes.length > 0,
    count: boxes.length,
    max_confidence: boxes.length ? boxes[0].confidence : 0,
    boxes,
    threshold
  };
}
