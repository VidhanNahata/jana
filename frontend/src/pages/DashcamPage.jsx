import React, { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { loadRoadDamage, detectRoadDamage } from '../utils/roadDamageDetector';
import { loadDetector, detectPotholes } from '../utils/potholeDetector';
import { PriorityBadge } from '../components/PriorityBadge';
import { useAuth } from '../context/AuthContext';
import { api } from '../utils/api';

// Road scan: the phone camera (handheld for now; mounted as a dashcam later) or
// a recorded video is scanned frame by frame on-device. Each road defect is tracked across frames
// so it is logged once, with its closest-approach photo, GPS position and
// speed. After the drive the citizen reviews each hazard: the server
// classifies it, routes it to the authority and sets its priority, and only
// the hazards the citizen ticks are submitted.

const FRAME_INTERVAL_MS = 60;
const HITS_TO_CONFIRM = 3;          // frames a track must be seen in before it is logged
const TRACK_TIMEOUT_MS = 1200;      // a track not seen for this long is closed
const TRACK_IOU = 0.2;
const SAME_HAZARD_METRES = 15;      // same class within this distance = the same defect
const NEARBY_METRES = 100;
const BLR_CORPS = [
  ['KA.BLR.CENTRAL', 'Bengaluru Central'], ['KA.BLR.NORTH', 'Bengaluru North'], ['KA.BLR.SOUTH', 'Bengaluru South'],
  ['KA.BLR.EAST', 'Bengaluru East'], ['KA.BLR.WEST', 'Bengaluru West']
];

function iou(a, b) {
  const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const inter = ix * iy;
  return inter / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter + 1e-9);
}

function metres(a, b) {
  const R = 6371000, rad = Math.PI / 180;
  const dLat = (b.lat - a.lat) * rad, dLon = (b.lon - a.lon) * rad;
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(a.lat * rad) * Math.cos(b.lat * rad) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
}

function snapshot(video) {
  const c = document.createElement('canvas');
  c.width = video.videoWidth;
  c.height = video.videoHeight;
  c.getContext('2d').drawImage(video, 0, 0);
  return new Promise((resolve) => c.toBlob(resolve, 'image/jpeg', 0.9));
}

function speak(text) {
  try {
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
  } catch { /* no speech support */ }
}

export function DashcamPage() {
  const { user } = useAuth();
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const posRef = useRef(null);
  const tracksRef = useRef([]);
  const hazardsRef = useRef([]);
  const runningRef = useRef(false);
  const voiceRef = useRef(true);
  const [mode, setMode] = useState(null);            // 'camera' | 'video' | null
  const [running, setRunning] = useState(false);
  const [boxes, setBoxes] = useState([]);
  const [hazards, setHazards] = useState([]);
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');
  const [gps, setGps] = useState(null);
  const [voice, setVoice] = useState(true);
  const [fps, setFps] = useState(0);
  const [manualLoc, setManualLoc] = useState({ lat: '', lon: '' });
  const [corporation, setCorporation] = useState('');
  voiceRef.current = voice;

  useEffect(() => () => stop(), []);

  const updateHazards = () => setHazards([...hazardsRef.current]);
  const latestRef = useRef([]);

  // Log what the camera sees right now, without waiting for 3 frames (handheld demos)
  const captureNow = () => {
    const d = latestRef.current[0];
    if (!d) {
      setStatus('Nothing detected in this frame. Point the camera at the damage and try again.');
      return;
    }
    const loc = currentLocation();
    const same = hazardsRef.current.find((h) => h.name === d.name &&
      (loc && h.location ? metres(h.location, loc) < SAME_HAZARD_METRES : performance.now() - (h.lastSeenMs || 0) < 3000));
    if (same) {
      Object.assign(same, { blobPromise: snapshot(videoRef.current), confidence: Math.max(same.confidence || 0, d.confidence), detection: d });
      updateHazards();
      setStatus(`${d.label} already logged here; photo updated`);
      return;
    }
    const h = { id: `${performance.now()}-manual`, name: d.name, label: d.label, subcategoryId: d.subcategoryId,
                seenAt: new Date(), frames: 1, selected: true, confidence: d.confidence,
                area: (d.x2 - d.x1) * (d.y2 - d.y1), location: currentLocation(), detection: d,
                videoTime: videoRef.current.currentTime, blobPromise: snapshot(videoRef.current) };
    hazardsRef.current.push(h);
    updateHazards();
    setStatus(`Captured: ${d.label}`);
  };

  const start = async (kind, file) => {
    setError('');
    setStatus('Loading road-damage model…');
    try {
      await Promise.all([loadRoadDamage(), loadDetector()]);
    } catch (err) {
      setError(`Could not load the road-damage model: ${err.message}`);
      return;
    }
    const video = videoRef.current;
    if (kind === 'camera') {
      if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
        setError('Camera needs HTTPS and a browser with camera support.');
        return;
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false
        });
        streamRef.current = stream;
        video.srcObject = stream;
      } catch (err) {
        setError(err.name === 'NotAllowedError' ? 'Camera permission denied.' : `Camera error: ${err.message}`);
        return;
      }
      if ('geolocation' in navigator) {
        posRef.current = null;
        streamRef.watchId = navigator.geolocation.watchPosition(
          (p) => {
            posRef.current = p;
            setGps({ lat: p.coords.latitude, lon: p.coords.longitude, speed: p.coords.speed });
          },
          () => {},
          { enableHighAccuracy: true, maximumAge: 1000 }
        );
      }
    } else {
      video.srcObject = null;
      video.src = URL.createObjectURL(file);
      video.onended = () => stop();
    }
    await video.play().catch(() => {});
    setMode(kind);
    runningRef.current = true;
    setRunning(true);
    setStatus(kind === 'camera' ? 'Scanning the road ahead…' : 'Scanning the video…');
    loop();
  };

  const stop = () => {
    runningRef.current = false;
    setRunning(false);
    streamRef.current?.getTracks?.().forEach((t) => t.stop());
    if (streamRef.watchId != null) navigator.geolocation.clearWatch(streamRef.watchId);
    streamRef.watchId = null;
    videoRef.current?.pause?.();
    setBoxes([]);
    setStatus(hazardsRef.current.length ? 'Drive finished. Review the hazards below.' : 'Stopped.');
  };

  const currentLocation = () => {
    const p = posRef.current;
    if (p) {
      return {
        lat: p.coords.latitude, lon: p.coords.longitude,
        speedKmh: p.coords.speed != null ? p.coords.speed * 3.6 : null, source: 'dashcam_gps'
      };
    }
    return null;
  };

  // Tracks: same class + overlapping box in consecutive frames = one defect
  const track = async (dets, now) => {
    const video = videoRef.current;
    const tracks = tracksRef.current;
    for (const d of dets) {
      const area = (d.x2 - d.x1) * (d.y2 - d.y1);
      let t = tracks.find((t) => t.classIndex === d.classIndex && iou(t.box, d) > TRACK_IOU);
      if (!t) {
        t = { classIndex: d.classIndex, hits: 0, best: null, logged: null };
        tracks.push(t);
      }
      t.box = d;
      t.hits += 1;
      t.lastSeen = now;
      // Keep the closest approach (largest box) as the evidence photo
      if (!t.best || area > t.best.area) {
        t.best = { area, confidence: d.confidence, detection: d, location: currentLocation(),
                   videoTime: video.currentTime, blobPromise: snapshot(video) };
      }
      if (t.hits >= HITS_TO_CONFIRM && !t.logged) {
        const loc = t.best.location;
        const same = loc && hazardsRef.current.find((h) =>
          h.name === d.name && h.location && metres(h.location, loc) < SAME_HAZARD_METRES);
        if (same) {
          t.logged = same;
        } else {
          const h = { id: `${now}-${d.classIndex}`, name: d.name, label: d.label, subcategoryId: d.subcategoryId,
                      seenAt: new Date(), frames: 0, selected: true };
          hazardsRef.current.push(h);
          t.logged = h;
          if (voiceRef.current) speak(`${d.label} ahead`);
        }
      }
      if (t.logged) {
        Object.assign(t.logged, {
          frames: t.logged.frames + 1,
          confidence: Math.max(t.logged.confidence || 0, t.best.confidence),
          area: t.best.area, location: t.best.location || t.logged.location, videoTime: t.best.videoTime, lastSeenMs: now,
          blobPromise: t.best.blobPromise, detection: t.best.detection
        });
      }
    }
    tracksRef.current = tracks.filter((t) => now - t.lastSeen < TRACK_TIMEOUT_MS);
  };

  const loop = async () => {
    let last = performance.now();
    while (runningRef.current) {
      const video = videoRef.current;
      if (video && video.readyState >= 2 && video.videoWidth && !video.paused) {
        try {
          // Potholes: our pothole model (the pretrained road-damage model misses Indian
          // potholes); cracks: the pretrained RDD2022 model
          const [cracks, potholes] = [await detectRoadDamage(video), await detectPotholes(video)];
          const dets = [...cracks, ...potholes.boxes.map(({ x1, y1, x2, y2, confidence }) => ({
            x1, y1, x2, y2, confidence, classIndex: 100, name: 'pothole', label: 'Pothole', subcategoryId: 'ROAD.POTHOLE'
          }))];
          const now = performance.now();
          setBoxes(dets);
          latestRef.current = [...dets].sort((a, b) => b.confidence - a.confidence);
          await track(dets, now);
          updateHazards();
          setFps(Math.round(1000 / Math.max(1, now - last)));
          last = now;
        } catch (err) {
          setError(`Detection failed: ${err.message}`);
          stop();
          return;
        }
      }
      await new Promise((r) => setTimeout(r, FRAME_INTERVAL_MS));
    }
  };

  const locationOf = (h) => {
    if (h.location) return h.location;
    const lat = parseFloat(manualLoc.lat), lon = parseFloat(manualLoc.lon);
    return Number.isFinite(lat) && Number.isFinite(lon) ? { lat, lon, source: 'manual' } : null;
  };

  const nearbyCount = (h) => {
    const loc = locationOf(h);
    if (!h.location || !loc) return null;
    return hazardsRef.current.filter((o) => o !== h && o.location && metres(o.location, loc) < NEARBY_METRES).length + 1;
  };

  const formFor = async (h, withImage) => {
    const loc = locationOf(h);
    const fd = new FormData();
    const when = h.seenAt.toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' });
    const speed = h.location?.speedKmh;
    const description = `${h.label} on the road, recorded by ${mode === 'video' ? 'video' : 'live phone camera'} scan on ${when}` +
      (loc ? ` near ${loc.lat.toFixed(5)}, ${loc.lon.toFixed(5)}` : '') +
      (speed != null ? ` while driving at about ${Math.round(speed)} km/h` : '') + '.';
    fd.append('description', description);
    fd.append('latitude', loc.lat);
    fd.append('longitude', loc.lon);
    if (corporation) fd.append('city_corporation_id', corporation);
    fd.append('dashcam_evidence', JSON.stringify({
      speed_kmh: speed ?? null, nearby_detections: nearbyCount(h), track_frames: h.frames, detector_class: h.name
    }));
    if (withImage) {
      const blob = await h.blobPromise;
      fd.append('images', new File([blob], `dashcam_${h.id}.jpg`, { type: 'image/jpeg' }));
      fd.append('live_detection', 'true');
    }
    return { fd, description, loc };
  };

  const analyse = async (h) => {
    if (!locationOf(h)) {
      h.error = 'No GPS for this hazard: enter the drive location above.';
      updateHazards();
      return;
    }
    h.busy = true;
    h.error = null;
    updateHazards();
    try {
      const { fd } = await formFor(h, true);
      h.analysis = await api.upload('/ai/analyze', fd);
      h.preview = h.preview || URL.createObjectURL(await h.blobPromise);
    } catch (err) {
      h.error = err.message;
    }
    h.busy = false;
    updateHazards();
  };

  const submit = async (h) => {
    const a = h.analysis;
    if (!a || a.status === 'out_of_scope') return;
    if (a.routing.missing_location_fields.includes('city_corporation') && !corporation) {
      h.error = 'Choose the Bengaluru city corporation above, then analyse again.';
      updateHazards();
      return;
    }
    h.busy = true;
    updateHazards();
    try {
      const { fd, loc } = await formFor(h, true);
      fd.append('location_source', loc.source || 'dashcam_gps');
      fd.append('captured_at', h.seenAt.toISOString());
      fd.append('analysis_id', a.analysis_id);
      if (a.complaint_draft) fd.append('complaint_text', `${a.complaint_draft.subject}\n\n${a.complaint_draft.body}`);
      h.result = await api.upload('/complaints', fd);
    } catch (err) {
      h.error = err.message;
    }
    h.busy = false;
    updateHazards();
  };

  const selected = hazards.filter((h) => h.selected && !h.result);
  const analyseAll = async () => { for (const h of selected) if (!h.analysis) await analyse(h); };
  const submitAll = async () => { for (const h of selected) if (h.analysis && !h.result) await submit(h); };
  const needGps = hazards.some((h) => !h.location);
  const fullscreen = running && mode === 'camera';

  return (
    <div className="container">
      <div className="mb-4">
        <h1>Road Scan</h1>
        <p className="text-secondary">
          Point your phone camera at the road, or open a recorded video. Potholes and road cracks are detected live
          on your phone, each one is logged once with a photo and GPS position, and you review them before anything
          is submitted. (Dashboard-mounted dashcam use will come later; never handle the phone while driving.)
        </p>
      </div>

      {!user && <div className="authenticity-banner mb-4">Please <Link to="/login">log in</Link> to report hazards.</div>}

      <div className="card mb-4">
        <div className="flex gap-2 items-center" style={{ flexWrap: 'wrap' }}>
          {!running ? (
            <>
              <button className="btn btn-primary" onClick={() => start('camera')}>📱 Start camera scan</button>
              <label className="btn btn-outline">
                📂 Open a video
                <input type="file" accept="video/*" hidden onChange={(e) => e.target.files[0] && start('video', e.target.files[0])} />
              </label>
            </>
          ) : (
            <button className="btn btn-outline" onClick={stop}>■ Stop</button>
          )}
          <label className="text-sm flex items-center gap-1">
            <input type="checkbox" checked={voice} onChange={(e) => setVoice(e.target.checked)} /> Voice alerts
          </label>
          {running && <span className="text-sm text-secondary">{fps} fps · {hazards.length} hazard(s)</span>}
          {mode === 'camera' && (
            <span className="text-sm text-secondary">
              📍 {gps ? `GPS ${gps.speed != null ? `· ${Math.round(gps.speed * 3.6)} km/h` : ''}` : 'Waiting for GPS…'}
            </span>
          )}
        </div>
        {status && <div className="text-sm mt-2">{status}</div>}
        {error && <div className="form-error text-sm mt-2">{error}</div>}

        {/* Camera scans go full screen (phone demo); videos play inline */}
        <div className={fullscreen ? 'live-camera-overlay' : 'scan-inline mt-2'} style={{ display: mode === 'video' || fullscreen ? undefined : 'none' }}>
          {fullscreen && (
            <div className="live-camera-header">
              <span className="live-camera-title"><span className="live-dot" /> Road Scan · {hazards.length} found</span>
              <button type="button" className="btn btn-sm btn-outline live-camera-close" onClick={stop}>■ Done</button>
            </div>
          )}
          <div className={fullscreen ? 'live-camera-frame' : ''}>
            <div className="live-camera-stage">
              <video ref={videoRef} autoPlay playsInline muted className="live-camera-video" />
              {boxes.map((b, i) => (
                <div key={i} className={`detection-box dashcam-${b.name}`}
                  style={{ left: `${b.x1 * 100}%`, top: `${b.y1 * 100}%`, width: `${(b.x2 - b.x1) * 100}%`, height: `${(b.y2 - b.y1) * 100}%` }}>
                  <span className="detection-label">{b.label} {Math.round(b.confidence * 100)}%</span>
                </div>
              ))}
            </div>
          </div>
          {fullscreen && (
            <div className="live-camera-status">
              <div>{error ? <span className="text-error">{error}</span> : status}</div>
              <div className="flex gap-2 items-center mt-2" style={{ justifyContent: 'center', flexWrap: 'wrap' }}>
                <button type="button" className="btn btn-primary" onClick={captureNow}>📸 Capture</button>
                <label className="text-sm flex items-center gap-1">
                  <input type="checkbox" checked={voice} onChange={(e) => setVoice(e.target.checked)} /> Voice
                </label>
                <span className="text-sm" style={{ opacity: 0.8 }}>
                  {fps} fps · 📍 {gps ? `GPS${gps.speed != null ? ` · ${Math.round(gps.speed * 3.6)} km/h` : ''}` : 'waiting for GPS'}
                </span>
              </div>
            </div>
          )}
        </div>
      </div>

      {hazards.length > 0 && (
        <div className="card mb-4">
          <h3 className="mb-2">Hazards found ({hazards.length})</h3>
          <p className="text-sm text-secondary mb-2">
            Each hazard is classified and routed by the server and gets a priority from its severity, size, road
            speed and how many defects are nearby. Untick anything that is wrong; only ticked hazards are submitted.
          </p>
          {needGps && (
            <div className="flex gap-2 mb-2" style={{ flexWrap: 'wrap' }}>
              <span className="text-sm">Video has no GPS. Drive location:</span>
              <input className="form-input" style={{ width: 130 }} placeholder="Latitude" value={manualLoc.lat}
                onChange={(e) => setManualLoc({ ...manualLoc, lat: e.target.value })} />
              <input className="form-input" style={{ width: 130 }} placeholder="Longitude" value={manualLoc.lon}
                onChange={(e) => setManualLoc({ ...manualLoc, lon: e.target.value })} />
            </div>
          )}
          <div className="flex gap-2 mb-2 items-center" style={{ flexWrap: 'wrap' }}>
            <span className="text-sm">In Bengaluru? City corporation:</span>
            <select className="form-select" style={{ width: 200 }} value={corporation} onChange={(e) => setCorporation(e.target.value)}>
              <option value="">Not Bengaluru / not sure</option>
              {BLR_CORPS.map(([id, n]) => <option key={id} value={id}>{n}</option>)}
            </select>
          </div>
          <div className="flex gap-2 mb-4">
            <button className="btn btn-outline" disabled={!user || running || !selected.length} onClick={analyseAll}>Classify selected</button>
            <button className="btn btn-primary" disabled={!user || running || !selected.some((h) => h.analysis)} onClick={submitAll}>Submit selected</button>
          </div>

          {hazards.map((h) => {
            const a = h.analysis;
            return (
              <div key={h.id} className="dashcam-hazard">
                <input type="checkbox" checked={h.selected} disabled={!!h.result}
                  onChange={(e) => { h.selected = e.target.checked; updateHazards(); }} />
                {h.preview ? <img src={h.preview} alt={h.label} /> : <div className="dashcam-thumb">{h.label}</div>}
                <div style={{ flex: 1 }}>
                  <strong>{h.label}</strong> <span className="text-sm text-tertiary">{Math.round((h.confidence || 0) * 100)}% · {h.frames} frames</span>
                  <div className="text-sm text-secondary">
                    {h.location ? `${h.location.lat.toFixed(5)}, ${h.location.lon.toFixed(5)}` : 'no GPS'}
                    {h.location?.speedKmh != null && ` · ${Math.round(h.location.speedKmh)} km/h`}
                    {h.videoTime != null && mode === 'video' && ` · at ${h.videoTime.toFixed(1)} s`}
                  </div>
                  {a && (
                    <div className="text-sm mt-1">
                      {a.status === 'out_of_scope' ? 'Not a civic problem' : <>
                        {a.classification.subcategory_name_en} → {a.routing.authority_name || 'authority pending confirmation'}{' '}
                        {a.priority_preview && <PriorityBadge priority={a.priority_preview.priority} />}
                        <div className="text-tertiary">{a.priority_preview?.factors.join(' · ')}</div>
                        {a.classification.review_reasons.length > 0 && (
                          <div className="text-tertiary">Review: {a.classification.review_reasons.join('; ')}</div>
                        )}
                      </>}
                    </div>
                  )}
                  {h.result && <div className="text-sm mt-1">✅ Submitted as <Link to={`/track/${h.result.ticket_id}`}>{h.result.ticket_id}</Link> ({h.result.priority})</div>}
                  {h.error && <div className="form-error text-sm">{h.error}</div>}
                </div>
                <div className="flex gap-1" style={{ flexDirection: 'column' }}>
                  {!h.result && <button className="btn btn-sm btn-outline" disabled={!user || h.busy || running} onClick={() => analyse(h)}>{a ? 'Re-check' : 'Classify'}</button>}
                  {!h.result && a && a.status !== 'out_of_scope' && (
                    <button className="btn btn-sm btn-primary" disabled={!user || h.busy || running} onClick={() => submit(h)}>Submit</button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
