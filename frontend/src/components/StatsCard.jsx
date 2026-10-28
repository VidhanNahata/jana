import React from 'react';

export function StatsCard({ icon, value, label, colorClass = "text-primary" }) {
  return (
    <div className="card stat-card">
      <div className={`stat-icon ${colorClass}`} style={{ backgroundColor: 'var(--color-surface-hover)' }}>
        {icon}
      </div>
      <div className="stat-content">
        <h3>{value}</h3>
        <p>{label}</p>
      </div>
    </div>
  );
}
