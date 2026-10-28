import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { api, assetUrl } from '../utils/api';
import { PriorityBadge } from '../components/PriorityBadge';
import { StatusBadge } from '../components/StatusBadge';
import { StatusTimeline } from '../components/StatusTimeline';
import { LoadingSpinner } from '../components/LoadingSpinner';

export function TrackPage() {
  const { ticketId: paramTicketId } = useParams();
  
  const [ticketId, setTicketId] = useState(paramTicketId || '');
  const [complaint, setComplaint] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (paramTicketId) {
      setTicketId(paramTicketId);
      handleSearch(paramTicketId);
    }
  }, [paramTicketId]);

  const handleSearch = async (idToSearch = ticketId) => {
    if (!idToSearch.trim()) return;
    
    setLoading(true);
    setError('');
    setComplaint(null);

    try {
      const data = await api.get(`/complaints/track/${idToSearch.trim()}`);
      setComplaint(data);
    } catch (err) {
      setError(err.message || 'Ticket not found. Please check the ID and try again.');
    } finally {
      setLoading(false);
    }
  };

  const onSubmit = (e) => {
    e.preventDefault();
    handleSearch(ticketId);
  };

  const copyTicketId = () => {
    if (complaint?.ticket_id) {
      navigator.clipboard.writeText(complaint.ticket_id);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="centered-page">
      <div className="card-container">
        <div style={{ textAlign: 'center', marginBottom: 'var(--space-8)' }}>
          <h1>Track Your Complaint</h1>
          <p className="text-secondary">Enter your ticket ID to check the current status</p>
        </div>

        <form onSubmit={onSubmit} style={{ display: 'flex', gap: 'var(--space-3)', marginBottom: 'var(--space-8)' }}>
          <input 
            type="text" 
            className="form-input" 
            placeholder="e.g. JS-2026-00001" 
            value={ticketId}
            onChange={(e) => setTicketId(e.target.value)}
            style={{ flex: 1, fontFamily: 'var(--font-mono)', letterSpacing: '0.5px' }}
          />
          <button type="submit" className="btn btn-primary" disabled={!ticketId.trim() || loading}>
            {loading ? 'Searching...' : 'Track'}
          </button>
        </form>

        {loading && <LoadingSpinner />}

        {error && (
          <div className="card" style={{ textAlign: 'center', padding: 'var(--space-8)', color: 'var(--color-error)' }}>
            <div style={{ fontSize: '32px', marginBottom: 'var(--space-3)' }}>😕</div>
            <p style={{ fontWeight: 500 }}>{error}</p>
            <p className="text-secondary" style={{ fontSize: '14px', marginTop: 'var(--space-2)' }}>
              Make sure you've entered the correct ticket ID
            </p>
          </div>
        )}

        {complaint && !loading && (
          <div className="card" style={{ animation: 'fadeIn 0.3s ease' }}>
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 'var(--space-6)' }}>
              <div>
                <p className="text-secondary" style={{ fontSize: '13px', marginBottom: '4px' }}>Ticket ID</p>
                <div 
                  onClick={copyTicketId}
                  style={{ 
                    fontFamily: 'var(--font-mono)', fontSize: '20px', fontWeight: 600,
                    cursor: 'pointer', color: 'var(--color-primary)',
                  }}
                  title="Click to copy"
                >
                  {complaint.ticket_id}
                  <span style={{ fontSize: '13px', color: 'var(--color-text-secondary)', marginLeft: '8px' }}>
                    {copied ? '✓ Copied' : '📋'}
                  </span>
                </div>
              </div>
              <PriorityBadge priority={complaint.priority} />
            </div>

            {/* Description */}
            <div style={{ marginBottom: 'var(--space-6)' }}>
              <p style={{ lineHeight: 1.6 }}>{complaint.description}</p>
            </div>

            {/* Images */}
            {complaint.images && complaint.images.length > 0 && (
              <div style={{ display: 'flex', gap: 'var(--space-3)', flexWrap: 'wrap', marginBottom: 'var(--space-6)' }}>
                {complaint.images.map((img, idx) => (
                  <img 
                    key={idx} 
                    src={assetUrl(img)} 
                    alt={`Complaint photo ${idx + 1}`}
                    style={{ 
                      width: '80px', height: '80px', objectFit: 'cover', 
                      borderRadius: 'var(--radius-sm)', border: '1px solid var(--color-border)' 
                    }} 
                  />
                ))}
              </div>
            )}

            {/* Details */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 'var(--space-2)', marginBottom: 'var(--space-6)' }}>
              <span className="badge" style={{ background: 'var(--color-primary-50)', color: 'var(--color-primary)' }}>
                {complaint.department}
              </span>
              <span className="badge" style={{ background: '#F1F5F9', color: '#475569' }}>
                {complaint.category}
              </span>
              <StatusBadge status={complaint.status} />
            </div>

            {/* Dates */}
            <div style={{ display: 'flex', gap: 'var(--space-6)', marginBottom: 'var(--space-6)', fontSize: '14px', color: 'var(--color-text-secondary)' }}>
              <div>
                <span style={{ fontWeight: 500 }}>Filed: </span>
                {new Date(complaint.created_at).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' })}
              </div>
              <div>
                <span style={{ fontWeight: 500 }}>Updated: </span>
                {new Date(complaint.updated_at).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' })}
              </div>
            </div>

            {/* Master Ticket Info */}
            {complaint.master_ticket_id && (
              <div style={{ 
                padding: 'var(--space-3) var(--space-4)', 
                background: '#EFF6FF', borderRadius: 'var(--radius-sm)',
                color: '#1E40AF', fontSize: '14px', marginBottom: 'var(--space-6)'
              }}>
                ℹ️ This report is linked to Master Ticket <strong>{complaint.master_ticket_id}</strong>
              </div>
            )}

            {/* Timeline */}
            <div style={{ borderTop: '1px solid var(--color-border)', paddingTop: 'var(--space-6)' }}>
              <h3 style={{ fontSize: '16px', fontWeight: 600, marginBottom: 'var(--space-4)' }}>Progress</h3>
              <StatusTimeline status={complaint.status} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
