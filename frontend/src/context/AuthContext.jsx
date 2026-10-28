import { createContext, useContext, useState, useEffect, useRef } from 'react';
import { api } from '../utils/api';

const AuthContext = createContext();

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('token') || null);
  const [loading, setLoading] = useState(true);
  const skipValidation = useRef(false);

  // Sync token to localStorage and validate on change
  useEffect(() => {
    if (token) {
      localStorage.setItem('token', token);
      // If login/register already set the user, skip the /me call
      if (skipValidation.current) {
        skipValidation.current = false;
        setLoading(false);
      } else {
        validateToken();
      }
    } else {
      localStorage.removeItem('token');
      setUser(null);
      setLoading(false);
    }
  }, [token]);

  const validateToken = async () => {
    try {
      const userData = await api.get('/auth/me');
      setUser(userData);
    } catch (error) {
      console.error('Token validation failed:', error);
      // Token is invalid/expired — clear it silently
      localStorage.removeItem('token');
      setToken(null);
      setUser(null);
    } finally {
      setLoading(false);
    }
  };

  const login = async (phoneOrEmail, password) => {
    const response = await api.post('/auth/login', {
      phone_or_email: phoneOrEmail,
      password
    });
    // Set user immediately from the response
    setUser(response.user);
    // Skip the useEffect validation since we already have the user
    skipValidation.current = true;
    // Write to localStorage BEFORE setting state to avoid race
    localStorage.setItem('token', response.token);
    setToken(response.token);
    return true;
  };

  const register = async (phoneOrEmail, password) => {
    const response = await api.post('/auth/register', {
      phone_or_email: phoneOrEmail,
      password
    });
    // Write to localStorage first
    localStorage.setItem('token', response.token);
    // Now fetch user details (register response only has id + token)
    try {
      const userData = await api.get('/auth/me');
      setUser(userData);
    } catch {
      setUser({ id: response.id, phone_or_email: phoneOrEmail });
    }
    skipValidation.current = true;
    setToken(response.token);
    return true;
  };

  const logout = () => {
    localStorage.removeItem('token');
    setToken(null);
    setUser(null);
  };

  const value = {
    user,
    token,
    isLoggedIn: !!user,
    isAdmin: user?.is_admin === true,
    loading,
    login,
    register,
    logout
  };

  return (
    <AuthContext.Provider value={value}>
      {!loading && children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
