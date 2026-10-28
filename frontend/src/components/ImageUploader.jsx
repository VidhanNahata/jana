import React, { useRef, useState } from 'react';
import { extractGPS } from '../utils/exif';

export function ImageUploader({ images, setImages, setLocation, onLiveScan }) {
  const cameraInputRef = useRef(null);
  const galleryInputRef = useRef(null);
  const [error, setError] = useState('');

  const handleFileChange = async (e, source) => {
    const files = Array.from(e.target.files);
    if (!files.length) return;
    
    if (images.length + files.length > 5) {
      setError('Maximum 5 images allowed');
      return;
    }
    setError('');

    const newImages = [];
    
    for (const file of files) {
      const url = URL.createObjectURL(file);
      newImages.push({ file, url });
      
      // Try to get GPS from file if it's from gallery
      if (source === 'gallery') {
        const gps = await extractGPS(file);
        if (gps) {
          setLocation({ ...gps, source: 'exif' });
        }
      }
    }

    // If from camera, get location from device
    if (source === 'camera' && 'geolocation' in navigator) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setLocation({
            lat: position.coords.latitude,
            lon: position.coords.longitude,
            source: 'camera'
          });
        },
        (err) => {
          console.warn('Geolocation failed:', err.message);
        }
      );
    }

    setImages(prev => [...prev, ...newImages]);
    
    // Reset inputs
    if (cameraInputRef.current) cameraInputRef.current.value = '';
    if (galleryInputRef.current) galleryInputRef.current.value = '';
  };

  const removeImage = (index) => {
    setImages(prev => {
      const newImages = [...prev];
      URL.revokeObjectURL(newImages[index].url);
      newImages.splice(index, 1);
      return newImages;
    });
  };

  return (
    <div className="mb-6">
      <input 
        type="file" 
        accept="image/*" 
        capture="environment" 
        ref={cameraInputRef} 
        style={{ display: 'none' }} 
        onChange={(e) => handleFileChange(e, 'camera')}
      />
      <input 
        type="file" 
        accept="image/*" 
        multiple 
        ref={galleryInputRef} 
        style={{ display: 'none' }} 
        onChange={(e) => handleFileChange(e, 'gallery')}
      />

      <div className="dropzone">
        {onLiveScan && (
          <button
            type="button"
            className="btn btn-accent btn-full mb-4"
            onClick={onLiveScan}
            disabled={images.length >= 5}
          >
            🎯 Live Pothole Scan — auto-detect with camera
          </button>
        )}
        <div className="flex justify-center gap-4 flex-wrap">
          <button 
            type="button" 
            className="btn btn-primary" 
            onClick={() => cameraInputRef.current?.click()}
            disabled={images.length >= 5}
          >
            📷 Take Photo
          </button>
          <button 
            type="button" 
            className="btn btn-secondary" 
            onClick={() => galleryInputRef.current?.click()}
            disabled={images.length >= 5}
          >
            📁 Choose from Gallery
          </button>
        </div>
        <p className="mt-4 text-secondary text-sm">
          {images.length}/5 images uploaded. Take a clear picture of the issue.
        </p>
        {error && <p className="form-error">{error}</p>}
      </div>

      {images.length > 0 && (
        <div className="thumbnail-row">
          {images.map((img, idx) => (
            <div key={idx} className="thumbnail-container">
              <img src={img.url} alt={`Upload ${idx}`} className="thumbnail" />
              <button 
                type="button" 
                className="remove-btn" 
                onClick={() => removeImage(idx)}
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
