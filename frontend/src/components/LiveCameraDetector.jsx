import React, { useEffect, useRef, useState } from 'react';
import { loadDetector, detectPotholes } from '../utils/potholeDetector';

// Short pause between on-device inferences to keep the phone responsive
const FRAME_INTERVAL_MS = 120;
// Require consecutive positive frames to avoid firing on a single noisy frame
const CONSECUTIVE_HITS_REQUIRED = 3;

function frameToBlob(video, maxWidth, quality) {
  const scale = maxWidth ? Math.min(1, maxWidth / video.videoWidth) : 1;
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', quality));
}

// Semi-transparent pothole fill drawn over the video (segmentation models only)
const MASK_RGBA = [236, 72, 153, 150];

function drawMasks(canvas, boxes) {
  const W = canvas.clientWidth;
  const H = canvas.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(W * dpr);
  canvas.height = Math.round(H * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  ctx.imageSmoothingEnabled = true; // softens the low-res mask edges when stretched
  for (const { mask, x1, y1, x2, y2 } of boxes) {
    if (!mask) continue;
    const tile = document.createElement('canvas');
    tile.width = mask.width;
    tile.height = mask.height;
    const img = new ImageData(mask.width, mask.height);
    mask.data.forEach((on, i) => {
      if (on) img.data.set(MASK_RGBA, i * 4);
    });
    tile.getContext('2d').putImageData(img, 0, 0);
    ctx.drawImage(tile, x1 * W, y1 * H, (x2 - x1) * W, (y2 - y1) * H);
  }
}

function getFreshPosition() {
  return new Promise((resolve) => {
    if (!('geolocation' in navigator)) return resolve(null);
    navigator.geolocation.getCurrentPosition(
      (pos) => resolve(pos),
      () => resolve(null),
      { enableHighAccuracy: true, timeout: 8000, maximumAge: 10000 }
    );
  });
}

export function LiveCameraDetector({ onDetected, onClose }) {
  const videoRef = useRef(null);
  const maskCanvasRef = useRef(null);
  const streamRef = useRef(null);
  const positionRef = useRef(null);
  const hitsRef = useRef(0);
  const doneRef = useRef(false);
  const onDetectedRef = useRef(onDetected);
  onDetectedRef.current = onDetected;
  const [boxes, setBoxes] = useState([]);
  const [status, setStatus] = useState('Starting camera...');
  const [error, setError] = useState('');
  const [hasGps, setHasGps] = useState(false);

  // Camera + GPS lifecycle
  useEffect(() => {
    let watchId = null;
    let cancelled = false;

    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      setError('Live camera needs HTTPS (or localhost) and a browser with camera support.');
      return;
    }

    navigator.mediaDevices
      .getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
      })
      .then((stream) => {
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        videoRef.current.srcObject = stream;
      })
      .catch((err) => {
        setError(err.name === 'NotAllowedError'
          ? 'Camera permission denied. Allow camera access and try again.'
          : `Could not start camera: ${err.message}`);
      });

    if ('geolocation' in navigator) {
      watchId = navigator.geolocation.watchPosition(
        (pos) => {
          positionRef.current = pos;
          setHasGps(true);
        },
        (err) => console.warn('Geolocation failed:', err.message),
        { enableHighAccuracy: true, maximumAge: 5000 }
      );
    }

    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((t) => t.stop());
      if (watchId !== null) navigator.geolocation.clearWatch(watchId);
    };
  }, []);

  // Detection loop: runs the model on-device, one frame at a time
  useEffect(() => {
    let timer = null;
    let stopped = false;

    const tick = async () => {
      if (stopped || doneRef.current) return;
      const video = videoRef.current;
      if (video && video.readyState >= 2 && video.videoWidth) {
        try {
          const result = await detectPotholes(video);
          if (stopped) return;

          setBoxes(result.boxes);
          if (result.detected) {
            hitsRef.current += 1;
            setStatus(`Pothole spotted (${Math.round(result.max_confidence * 100)}%) — hold steady...`);
          } else {
            hitsRef.current = 0;
            setStatus('Scanning for potholes...');
          }

          if (hitsRef.current >= CONSECUTIVE_HITS_REQUIRED) {
            doneRef.current = true;
            await handleCapture(result);
            return;
          }
        } catch (err) {
          if (stopped) return;
          setError(`Detection failed: ${err.message}`);
          return;
        }
      }
      timer = setTimeout(tick, FRAME_INTERVAL_MS);
    };

    // Download + initialise the model before the first frame (cached after first visit)
    setStatus('Loading pothole AI model...');
    loadDetector()
      .then(() => {
        if (stopped) return;
        setStatus('Point the camera at the road surface...');
        timer = setTimeout(tick, FRAME_INTERVAL_MS);
      })
      .catch((err) => {
        if (!stopped) setError(`Could not load the detection model: ${err.message}`);
      });
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, []);

  useEffect(() => {
    if (maskCanvasRef.current) drawMasks(maskCanvasRef.current, boxes);
  }, [boxes]);

  const handleCapture = async (detection) => {
    setStatus('Pothole confirmed! Capturing photo and location...');
    const capturedAt = new Date();
    const blob = await frameToBlob(videoRef.current, null, 0.92);
    const file = new File([blob], `pothole_${capturedAt.getTime()}.jpg`, { type: 'image/jpeg' });

    const pos = positionRef.current || await getFreshPosition();
    const location = pos
      ? { lat: pos.coords.latitude, lon: pos.coords.longitude, source: 'live_camera' }
      : null;

    onDetectedRef.current({ file, location, capturedAt, detection });
  };

  return (
    <div className="live-camera-overlay" role="dialog" aria-label="Live pothole detection">
      <div className="live-camera-header">
        <span className="live-camera-title">
          <span className="live-dot" /> Live Pothole Scan
        </span>
        <button type="button" className="btn btn-sm btn-outline live-camera-close" onClick={onClose}>
          ✕ Close
        </button>
      </div>

      <div className="live-camera-frame">
        <div className="live-camera-stage">
          <video ref={videoRef} autoPlay playsInline muted className="live-camera-video" />
          <canvas ref={maskCanvasRef} className="detection-masks" />
          {boxes.map((b, i) => (
            <div
              key={i}
              className={b.mask ? 'detection-box detection-box-seg' : 'detection-box'}
              style={{
                left: `${b.x1 * 100}%`,
                top: `${b.y1 * 100}%`,
                width: `${(b.x2 - b.x1) * 100}%`,
                height: `${(b.y2 - b.y1) * 100}%`
              }}
            >
              <span className="detection-label">pothole {Math.round(b.confidence * 100)}%</span>
            </div>
          ))}
        </div>
      </div>

      <div className="live-camera-status">
        {error ? <span className="text-error">{error}</span> : status}
        <div className="text-sm mt-1" style={{ opacity: 0.8 }}>
          📍 {hasGps ? 'GPS locked' : 'Waiting for GPS...'}
        </div>
      </div>
    </div>
  );
}
