import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export function Navbar() {
  const { user, isLoggedIn, isAdmin, logout } = useAuth();
  const [menuOpen, setMenuOpen] = useState(false);
  const location = useLocation();

  const isActive = (path) => location.pathname === path;

  return (
    <nav className="navbar">
      <div className="container">
        <Link to="/" className="navbar-brand">
          🏛️ JanaSutra
        </Link>

        {/* Desktop Nav */}
        <div className="navbar-nav">
          <Link to="/" className={`nav-link ${isActive('/') ? 'active' : ''}`}>Report</Link>
          <Link to="/scan" className={`nav-link ${isActive('/scan') ? 'active' : ''}`}>Road Scan</Link>
          <Link to="/track" className={`nav-link ${isActive('/track') ? 'active' : ''}`}>Track</Link>
          
          {isLoggedIn && (
            <Link to="/my-complaints" className={`nav-link ${isActive('/my-complaints') ? 'active' : ''}`}>
              My Complaints
            </Link>
          )}
          
          {isAdmin && (
            <Link to="/admin" className={`nav-link ${isActive('/admin') ? 'active' : ''}`}>
              Admin
            </Link>
          )}

          {isLoggedIn ? (
            <div className="flex items-center gap-4">
              <div className="user-avatar" title={user.phone_or_email}>
                {user.phone_or_email ? user.phone_or_email.charAt(0).toUpperCase() : 'U'}
              </div>
              <button className="btn btn-outline btn-sm" onClick={logout}>Logout</button>
            </div>
          ) : (
            <Link to="/login" className="btn btn-primary btn-sm">Login</Link>
          )}
        </div>

        {/* Mobile Hamburger */}
        <button 
          className="hamburger-btn" 
          onClick={() => setMenuOpen(!menuOpen)}
        >
          {menuOpen ? '✕' : '☰'}
        </button>

        {/* Mobile Menu */}
        <div className={`mobile-menu ${menuOpen ? 'open' : ''}`}>
          <div className="flex justify-end mb-4">
            <button className="hamburger-btn" onClick={() => setMenuOpen(false)}>✕</button>
          </div>
          <Link to="/" className="nav-link" onClick={() => setMenuOpen(false)}>Report an Issue</Link>
          <Link to="/track" className="nav-link" onClick={() => setMenuOpen(false)}>Track Status</Link>
          
          {isLoggedIn && (
            <Link to="/my-complaints" className="nav-link" onClick={() => setMenuOpen(false)}>My Complaints</Link>
          )}
          
          {isAdmin && (
            <Link to="/admin" className="nav-link" onClick={() => setMenuOpen(false)}>Admin Dashboard</Link>
          )}

          <div className="mt-auto">
            {isLoggedIn ? (
              <button className="btn btn-outline btn-full" onClick={() => { logout(); setMenuOpen(false); }}>
                Logout
              </button>
            ) : (
              <Link to="/login" className="btn btn-primary btn-full" onClick={() => setMenuOpen(false)}>
                Login / Register
              </Link>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}
