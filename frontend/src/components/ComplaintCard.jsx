import React, { useState } from 'react';
import { PriorityBadge } from './PriorityBadge';
import { StatusBadge } from './StatusBadge';
import { assetUrl } from '../utils/api';

const FAKE_LABELS = {
  ai_generated: 'AI-generated photo',
  screen_photo: 'Photo of a screen',
  downloaded: 'Downloaded / screenshot',
};

export function ComplaintCard({ complaint, onClick, onDelete }) {
  const [confirming, setConfirming] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const handleDelete = async (e) => {
    e.stopPropagation(); // Prevent card click
    if (!confirming) {
      setConfirming(true);
      setTimeout(() => setConfirming(false), 3000); // Auto-cancel after 3s
      return;
    }
    setDeleting(true);
    try {
      await onDelete(complaint.id);
    } catch (err) {
      console.error('Delete failed:', err);
      setDeleting(false);
      setConfirming(false);
    }
  };

  const cancelDelete = (e) => {
    e.stopPropagation();
    setConfirming(false);
  };

  return (
    <div className="card card-interactive complaint-card" onClick={onClick} style={{ opacity: deleting ? 0.4 : 1, transition: 'opacity 0.3s' }}>
      <div className="complaint-card-header mb-2">
        <span className="ticket-id">{complaint.ticket_id}</span>
        <PriorityBadge priority={complaint.priority} />
      </div>
      
      {complaint.authenticity?.flagged && (
        <div className="authenticity-flag mb-2" title={complaint.authenticity.reasons.join('\n')}>
          ⚠️ Suspected fake: {FAKE_LABELS[complaint.authenticity.verdict] || complaint.authenticity.verdict}
        </div>
      )}

      {complaint.civic_ai && (complaint.civic_ai.review_reasons.length > 0
        || ['registry_unconfirmed', 'unresolved'].includes(complaint.civic_ai.department_basis)) && (
        <div className="ai-review-flag mb-2" title={[
          ...complaint.civic_ai.review_reasons,
          ...(['registry_unconfirmed', 'unresolved'].includes(complaint.civic_ai.department_basis)
            ? ['Responsible authority not confirmed'] : []),
        ].join('\n')}>
          🔎 Review: {complaint.civic_ai.subcategory_name} ({Math.round(complaint.civic_ai.confidence * 100)}%)
          {complaint.civic_ai.consistency === 'contradiction' && ' · photo/text mismatch'}
        </div>
      )}

      <div className="complaint-card-body">
        {complaint.images && complaint.images.length > 0 && (
          <div className="mb-2" style={{ height: '100px', overflow: 'hidden', borderRadius: 'var(--radius-sm)' }}>
            <img 
              src={assetUrl(complaint.images[0])} 
              alt="Complaint" 
              style={{ width: '100%', height: '100%', objectFit: 'cover' }} 
            />
          </div>
        )}
        <p className="complaint-card-desc">{complaint.description}</p>
      </div>

      <div className="complaint-card-footer mt-4">
        <div>
          <span className="text-secondary" style={{ marginRight: '8px' }}>{complaint.department}</span>
          <StatusBadge status={complaint.status} />
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span>{new Date(complaint.created_at || Date.now()).toLocaleDateString()}</span>
          {onDelete && !confirming && (
            <button
              onClick={handleDelete}
              style={{
                background: 'none', border: 'none', cursor: 'pointer',
                fontSize: '14px', padding: '4px 6px', borderRadius: '4px',
                opacity: 0.5, transition: 'opacity 0.2s'
              }}
              onMouseEnter={e => e.target.style.opacity = 1}
              onMouseLeave={e => e.target.style.opacity = 0.5}
              title="Delete complaint"
            >
              🗑️
            </button>
          )}
          {onDelete && confirming && (
            <div style={{ display: 'flex', gap: '4px' }} onClick={e => e.stopPropagation()}>
              <button
                onClick={handleDelete}
                disabled={deleting}
                style={{
                  background: 'var(--color-error)', color: 'white', border: 'none',
                  fontSize: '12px', padding: '3px 8px', borderRadius: '4px',
                  cursor: 'pointer', fontWeight: 500
                }}
              >
                {deleting ? '...' : 'Delete'}
              </button>
              <button
                onClick={cancelDelete}
                style={{
                  background: 'var(--color-surface-hover)', border: 'none',
                  fontSize: '12px', padding: '3px 8px', borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                Cancel
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
