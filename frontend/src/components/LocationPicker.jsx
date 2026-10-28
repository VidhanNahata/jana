import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Marker, useMapEvents } from 'react-leaflet';
import L from 'leaflet';

// Fix Leaflet's default icon path issues in React
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

function LocationMarker({ position, setPosition }) {
  useMapEvents({
    click(e) {
      setPosition({ lat: e.latlng.lat, lon: e.latlng.lng, source: 'manual' });
    },
  });

  return position ? <Marker position={[position.lat, position.lon]} /> : null;
}

export function LocationPicker({ location, setLocation }) {
  const [showMap, setShowMap] = useState(false);
  const defaultCenter = [12.9716, 77.5946]; // Bengaluru

  const handleLocate = () => {
    if ('geolocation' in navigator) {
      navigator.geolocation.getCurrentPosition(
        (pos) => setLocation({ lat: pos.coords.latitude, lon: pos.coords.longitude, source: 'gps' }),
        () => setShowMap(true)
      );
    } else {
      setShowMap(true);
    }
  };

  return (
    <div className="mb-6">
      <label className="form-label">Location</label>
      
      {location ? (
        <div>
          <div className="map-container mb-2">
            <MapContainer 
              center={[location.lat, location.lon]} 
              zoom={15} 
              style={{ height: '100%', width: '100%' }}
            >
              <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
              <LocationMarker position={location} setPosition={setLocation} />
            </MapContainer>
          </div>
          <p className="text-sm text-secondary flex justify-between items-center">
            <span>📍 {location.lat.toFixed(4)}°N, {location.lon.toFixed(4)}°E</span>
            <span className="badge badge-submitted">From {(location.source || 'unknown').replace('_', ' ')}</span>
          </p>
        </div>
      ) : (
        <div className="card text-center" style={{ padding: 'var(--space-4)' }}>
          <p className="mb-4 text-secondary">No location detected</p>
          <div className="flex justify-center gap-4">
            <button type="button" className="btn btn-secondary btn-sm" onClick={handleLocate}>
              📍 Use Current Location
            </button>
            <button type="button" className="btn btn-outline btn-sm" onClick={() => setShowMap(true)}>
              🗺️ Set Manually
            </button>
          </div>
          
          {showMap && (
            <div className="map-container mt-4">
              <MapContainer 
                center={defaultCenter} 
                zoom={12} 
                style={{ height: '100%', width: '100%' }}
              >
                <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                <LocationMarker position={location} setPosition={setLocation} />
              </MapContainer>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
