import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { ImageUploader } from '../components/ImageUploader';
import { LiveCameraDetector } from '../components/LiveCameraDetector';
import { LocationPicker } from '../components/LocationPicker';
import { PriorityBadge } from '../components/PriorityBadge';
import { AiReview, canSubmit } from '../components/AiReview';
import { api } from '../utils/api';
import { checkPhoto } from '../utils/authenticityCheck';

export function ReportPage() {
  const [images, setImages] = useState([]);
  const [description, setDescription] = useState('');
  const [location, setLocation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [showLiveScan, setShowLiveScan] = useState(false);
  const [liveDetection, setLiveDetection] = useState(null);
  // Authenticity result per image url: { verdict, flagged, reasons } or 'checking'
  const [photoChecks, setPhotoChecks] = useState({});
  // Civic AI review step (POST /api/ai/analyze), done before submitting
  const [locality, setLocality] = useState('');
  const [pinCode, setPinCode] = useState('');
  const [rural, setRural] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [analysisFailed, setAnalysisFailed] = useState(false);
  const [choices, setChoices] = useState({});
  const [draftText, setDraftText] = useState('');
  const [analyzing, setAnalyzing] = useState(false);
  const [taxonomy, setTaxonomy] = useState({});

  useEffect(() => {
    api.get('/ai/taxonomy').then(t => {
      const map = {};
      t.categories.forEach(c => c.subcategories.forEach(sub => {
        map[sub.id] = { ...sub, category_name_en: c.name_en };
      }));
      setTaxonomy(map);
    }).catch(() => {});
  }, []);

  // Any change to what was analysed makes the analysis stale
  useEffect(() => {
    setAnalysis(null);
    setAnalysisFailed(false);
    setChoices({});
  }, [images, description, location, locality, pinCode, rural]);

  const runAnalysis = async (nextChoices = choices, previous = null) => {
    setAnalyzing(true);
    setError('');
    try {
      const formData = new FormData();
      formData.append('description', description);
      if (location) {
        formData.append('latitude', location.lat);
        formData.append('longitude', location.lon);
      }
      if (locality.trim()) formData.append('locality', locality.trim());
      if (pinCode.trim()) formData.append('pin_code', pinCode.trim());
      if (rural) formData.append('rural', 'true');
      if (nextChoices.confirmed_subcategory_id) formData.append('confirmed_subcategory_id', nextChoices.confirmed_subcategory_id);
      if (nextChoices.city_corporation_id) formData.append('city_corporation_id', nextChoices.city_corporation_id);
      formData.append('draft_language', nextChoices.draft_language || 'en');
      if (previous) {
        formData.append('analysis_id', previous.analysis_id);
      } else {
        images.forEach(img => formData.append('images', img.file));
        if (liveDetection) formData.append('live_detection', 'true');
      }
      const result = await api.upload('/ai/analyze', formData);
      setAnalysis(result);
      setChoices(nextChoices);
      setDraftText(result.complaint_draft ? `${result.complaint_draft.subject}\n\n${result.complaint_draft.body}` : '');
    } catch (err) {
      if (previous && /expired/i.test(err.message || '')) return runAnalysis(nextChoices, null);
      setAnalysisFailed(true);
      setError(`AI check unavailable (${err.message}). You can still submit the report.`);
    } finally {
      setAnalyzing(false);
    }
  };

  const handleChoose = (patch, { rerun = true } = {}) => {
    const next = { ...choices, ...patch };
    if ('confirmed_subcategory_id' in patch) next.confirmed_authority_id = null;  // authority depends on the issue
    setChoices(next);
    if (rerun) runAnalysis(next, analysis);
  };

  useEffect(() => {
    images.forEach((img) => {
      if (photoChecks[img.url]) return;
      setPhotoChecks(prev => ({ ...prev, [img.url]: 'checking' }));
      checkPhoto(img.file, { live: !!img.live })
        .then(result => setPhotoChecks(prev => ({ ...prev, [img.url]: result })))
        .catch(() => setPhotoChecks(prev => ({ ...prev, [img.url]: { verdict: 'unknown', flagged: false, reasons: [] } })));
    });
  }, [images]);

  const flaggedPhotos = images.map(img => photoChecks[img.url]).filter(c => c && c !== 'checking' && c.flagged);

  const handleLiveDetected = ({ file, location: detectedLocation, capturedAt, detection }) => {
    setShowLiveScan(false);
    setImages(prev => [...prev, { file, url: URL.createObjectURL(file), live: true }]);
    if (detectedLocation) setLocation(detectedLocation);

    const when = capturedAt.toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' });
    const count = detection.count === 1 ? '1 pothole' : `${detection.count} potholes`;
    const where = detectedLocation
      ? ` near ${detectedLocation.lat.toFixed(5)}, ${detectedLocation.lon.toFixed(5)}`
      : '';
    setDescription(`Pothole on road detected by live camera${where} on ${when}. ${count} visible.`.slice(0, 500));
    setLiveDetection({ capturedAt, confidence: detection.max_confidence, count: detection.count, hasLocation: !!detectedLocation });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!images.length || !description || !location) {
      setError('Please provide an image, description, and location.');
      return;
    }
    if (!analysis && !analysisFailed) {
      runAnalysis({}, null);
      return;
    }
    if (!canSubmit(analysis, choices)) return;

    setLoading(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('description', description);
      formData.append('latitude', location.lat);
      formData.append('longitude', location.lon);
      formData.append('location_source', location.source || 'manual');
      if (liveDetection) {
        formData.append('captured_at', liveDetection.capturedAt.toISOString());
        formData.append('live_detection', 'true');
      }
      
      images.forEach((img) => {
        formData.append('images', img.file);
      });
      if (locality.trim()) formData.append('locality', locality.trim());
      if (pinCode.trim()) formData.append('pin_code', pinCode.trim());
      if (rural) formData.append('rural', 'true');
      if (analysis) {
        formData.append('analysis_id', analysis.analysis_id);
        if (choices.confirmed_subcategory_id) formData.append('confirmed_subcategory_id', choices.confirmed_subcategory_id);
        if (choices.city_corporation_id) formData.append('city_corporation_id', choices.city_corporation_id);
        if (choices.confirmed_authority_id) formData.append('confirmed_authority_id', choices.confirmed_authority_id);
        if (draftText.trim()) formData.append('complaint_text', draftText.trim());
      }

      const response = await api.upload('/complaints', formData);
      setResult(response);
    } catch (err) {
      setError(err.message || 'Failed to submit report. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const copyTicketId = () => {
    if (result?.ticket_id) {
      navigator.clipboard.writeText(result.ticket_id);
      alert('Ticket ID copied to clipboard!');
    }
  };

  const handleReset = () => {
    setImages([]);
    setDescription('');
    setLocation(null);
    setLiveDetection(null);
    setResult(null);
    setError('');
    setLocality('');
    setPinCode('');
    setRural(false);
  };

  if (result) {
    return (
      <div className="centered-page">
        <div className="card card-container text-center">
          <div className="success-checkmark">✓</div>
          <h2>Report Submitted Successfully!</h2>
          <p className="text-secondary mb-6">Thank you for helping improve Bengaluru.</p>
          
          <div className="card bg-light mb-6" style={{ backgroundColor: 'var(--color-surface-hover)' }}>
            <p className="text-sm text-secondary mb-2">Your Ticket ID</p>
            <div 
              className="ticket-id mb-4" 
              style={{ fontSize: '24px', color: 'var(--color-primary)', cursor: 'pointer' }}
              onClick={copyTicketId}
              title="Click to copy"
            >
              {result.ticket_id} 📋
            </div>
            
            <div className="flex justify-center gap-4 flex-wrap">
              {result.category && <span className="badge badge-submitted">{result.category}</span>}
              {result.department && <span className="badge badge-submitted">{result.department}</span>}
              {['registry_unconfirmed', 'unresolved'].includes(result.department_basis) && (
                <span className="badge badge-submitted">authority to be confirmed by officials</span>
              )}
              {result.priority && <PriorityBadge priority={result.priority} />}
            </div>
          </div>

          <div className="flex flex-col gap-4">
            <Link to={`/track/${result.ticket_id}`} className="btn btn-primary btn-full">
              Track your complaint →
            </Link>
            <button onClick={handleReset} className="btn btn-outline btn-full">
              Report another issue
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="centered-page">
      <div className="card card-container">
        <div className="mb-6 text-center">
          <h1>Report an Issue</h1>
          <p className="text-secondary">Help improve Bengaluru — snap a photo and describe the problem</p>
        </div>

        <form onSubmit={handleSubmit}>
          <ImageUploader
            images={images}
            setImages={setImages}
            setLocation={setLocation}
            onLiveScan={() => setShowLiveScan(true)}
          />

          {flaggedPhotos.length > 0 && (
            <div className="authenticity-banner mb-6">
              <strong>⚠️ This photo may not be a real, live photo of the issue</strong>
              <ul className="text-sm">
                {[...new Set(flaggedPhotos.flatMap(c => c.reasons))].map(r => <li key={r}>{r}</li>)}
              </ul>
              <div className="text-sm text-secondary">
                Please take a photo of the actual location with your camera. You can still submit,
                but the report will be marked for verification by officials.
              </div>
            </div>
          )}

          {liveDetection && (
            <div className="detection-banner mb-6">
              <strong>🎯 Pothole auto-detected</strong>
              <div className="text-sm">
                {Math.round(liveDetection.confidence * 100)}% confidence · captured{' '}
                {liveDetection.capturedAt.toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' })}
                {liveDetection.hasLocation
                  ? ' · location from device GPS'
                  : ' · GPS unavailable, please set the location below'}
              </div>
              <div className="text-sm text-secondary">Review the details below, then submit.</div>
            </div>
          )}
          
          <div className="form-group mb-6">
            <label className="form-label">Description</label>
            <input 
              type="text" 
              className="form-input" 
              placeholder="Briefly describe the issue..." 
              value={description}
              onChange={(e) => setDescription(e.target.value.slice(0, 500))}
            />
            <div className="flex justify-end mt-1">
              <span className="text-sm text-tertiary">{description.length}/500</span>
            </div>
          </div>

          <LocationPicker location={location} setLocation={setLocation} />

          <div className="flex gap-4 mb-4 flex-wrap">
            <div className="form-group" style={{ flex: '2 1 180px' }}>
              <label className="form-label">Locality / landmark <span className="text-tertiary">(helps routing)</span></label>
              <input type="text" className="form-input" placeholder="e.g. Jayanagar 4th Block"
                value={locality} onChange={(e) => setLocality(e.target.value.slice(0, 120))} />
            </div>
            <div className="form-group" style={{ flex: '1 1 110px' }}>
              <label className="form-label">PIN code</label>
              <input type="text" inputMode="numeric" className="form-input" placeholder="560041"
                value={pinCode} onChange={(e) => setPinCode(e.target.value.replace(/\D/g, '').slice(0, 6))} />
            </div>
          </div>
          <label className="ai-option mb-6">
            <input type="checkbox" checked={rural} onChange={(e) => setRural(e.target.checked)} />
            <span className="text-sm">This is in a village (Gram Panchayat area)</span>
          </label>

          {analysis && (
            <AiReview
              analysis={analysis}
              taxonomy={taxonomy}
              choices={choices}
              onChoose={handleChoose}
              draftText={draftText}
              setDraftText={setDraftText}
              busy={analyzing}
            />
          )}

          {error && <div className="form-error mb-4">{error}</div>}

          <button 
            type="submit" 
            className="btn btn-accent btn-full" 
            style={{ fontSize: '16px', padding: '16px' }}
            disabled={!images.length || !description || !location || loading || analyzing
              || (analysis && !canSubmit(analysis, choices))}
          >
            {loading ? 'Submitting...'
              : analyzing ? 'Checking photo and description…'
              : (!analysis && !analysisFailed) ? 'Check with AI →'
              : analysis?.status === 'out_of_scope' ? 'Not a civic issue'
              : 'Submit Report →'}
          </button>
          {analysis && (
            <button type="button" className="btn btn-outline btn-full mt-2" disabled={analyzing}
              onClick={() => runAnalysis({}, null)}>
              Re-check
            </button>
          )}
        </form>
      </div>

      {showLiveScan && (
        <LiveCameraDetector
          onDetected={handleLiveDetected}
          onClose={() => setShowLiveScan(false)}
        />
      )}
    </div>
  );
}
