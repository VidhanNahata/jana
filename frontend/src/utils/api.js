// Empty in dev (Vite proxies /api); set VITE_API_URL to the backend origin in production
export const API_ORIGIN = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '');
const BASE_URL = `${API_ORIGIN}/api`;

// Backend returns upload paths like /uploads/x.jpg — resolve them against the API origin
export function assetUrl(path) {
  return path && path.startsWith('/') ? `${API_ORIGIN}${path}` : path;
}

function getHeaders(isFormData = false) {
  const headers = {};
  const token = localStorage.getItem('token');
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  if (!isFormData) {
    headers['Content-Type'] = 'application/json';
  }
  return headers;
}

async function handleResponse(response) {
  if (!response.ok) {
    let errorMsg = `Server error (${response.status})`;
    try {
      const text = await response.text();
      try {
        const errorData = JSON.parse(text);
        if (typeof errorData.detail === 'string') {
          errorMsg = errorData.detail;
        } else if (Array.isArray(errorData.detail)) {
          errorMsg = errorData.detail.map(e => e.msg || JSON.stringify(e)).join(', ');
        } else if (errorData.detail && errorData.detail.message) {
          errorMsg = errorData.detail.message;
        } else if (errorData.message) {
          errorMsg = errorData.message;
        }
      } catch {
        if (text && text.trim().length > 0 && text.length < 200) {
          errorMsg = text.trim();
        }
      }
    } catch {
      // Response read failed
    }
    throw new Error(errorMsg);
  }
  // 204 No Content — nothing to parse
  if (response.status === 204) {
    return null;
  }
  // Some responses may have empty body
  const text = await response.text();
  if (!text) return null;
  return JSON.parse(text);
}

export const api = {
  get: async (url) => {
    const response = await fetch(`${BASE_URL}${url}`, {
      method: 'GET',
      headers: getHeaders()
    });
    return handleResponse(response);
  },
  post: async (url, body) => {
    const response = await fetch(`${BASE_URL}${url}`, {
      method: 'POST',
      headers: getHeaders(),
      body: JSON.stringify(body)
    });
    return handleResponse(response);
  },
  patch: async (url, body) => {
    const response = await fetch(`${BASE_URL}${url}`, {
      method: 'PATCH',
      headers: getHeaders(),
      body: JSON.stringify(body)
    });
    return handleResponse(response);
  },
  delete: async (url) => {
    const response = await fetch(`${BASE_URL}${url}`, {
      method: 'DELETE',
      headers: getHeaders()
    });
    return handleResponse(response);
  },
  upload: async (url, formData) => {
    const response = await fetch(`${BASE_URL}${url}`, {
      method: 'POST',
      headers: getHeaders(true),
      body: formData
    });
    return handleResponse(response);
  }
};
