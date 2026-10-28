import React from 'react';

export function StatusTimeline({ status, history }) {
  // history is array of { status, timestamp }
  // Provide defaults if not available
  const steps = [
    { key: 'submitted', label: 'Submitted' },
    { key: 'verified', label: 'Verified' },
    { key: 'in_progress', label: 'In Progress' },
    { key: 'resolved', label: 'Resolved' }
  ];

  const currentStatusIndex = steps.findIndex(s => s.key === (status || 'submitted').toLowerCase().replace(' ', '_'));
  const activeIndex = currentStatusIndex === -1 ? 0 : currentStatusIndex;

  return (
    <div className="timeline">
      {steps.map((step, idx) => {
        const isActive = idx <= activeIndex;
        return (
          <div key={step.key} className={`timeline-item ${isActive ? 'active' : ''}`}>
            <div className="timeline-dot"></div>
            <div className="timeline-content">
              <h4>{step.label}</h4>
              {/* Optional: Add dates here if history contains them */}
            </div>
          </div>
        );
      })}
    </div>
  );
}
