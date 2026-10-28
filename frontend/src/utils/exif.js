import exifr from 'exifr';

export async function extractGPS(file) {
  try {
    const { latitude, longitude } = await exifr.gps(file);
    if (latitude && longitude) return { lat: latitude, lon: longitude };
    return null;
  } catch {
    return null;
  }
}
