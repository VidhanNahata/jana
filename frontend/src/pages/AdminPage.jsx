import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { api } from '../utils/api';
import { StatsCard } from '../components/StatsCard';
import { ComplaintCard } from '../components/ComplaintCard';
import { PriorityBadge } from '../components/PriorityBadge';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import L from 'leaflet';

// Fix Leaflet default marker icons in bundled builds
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

// Priority-colored markers
function createPriorityIcon(priority) {
  const colors = { P1: '#DC2626', P2: '#EA580C', P3: '#CA8A04', P4: '#16A34A', P5: '#64748B' };
  const color = colors[priority] || '#64748B';
  return L.divIcon({
    className: 'priority-marker',
    html: `<div style="
      width: 14px; height: 14px; border-radius: 50%;
      background: ${color}; border: 2px solid white;
      box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    "></div>`,
    iconSize: [14, 14],
    iconAnchor: [7, 7],
  });
}

export function AdminPage() {
  const { isAdmin, isLoggedIn, loading: authLoading } = useAuth();
  const navigate = useNavigate();

  const [stats, setStats] = useState(null);
  const [complaints, setComplaints] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState({ dept: '', priority: '', status: '', authenticity: '' });
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    if (!authLoading && !isAdmin) {
      navigate('/login');
    } else if (isAdmin) {
      fetchDashboardData();
    }
  }, [isAdmin, authLoading]);

  const fetchDashboardData = async () => {
    try {
      const [statsData, complaintsData] = await Promise.all([
        api.get('/admin/stats'),
        api.get('/admin/complaints?limit=100')
      ]);
      setStats(statsData);
      setComplaints(complaintsData.complaints || []);
    } catch (err) {
      console.error('Dashboard fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const filteredComplaints = complaints.filter(c => {
    if (filter.dept && c.department !== filter.dept) return false;
    if (filter.priority && c.priority !== filter.priority) return false;
    if (filter.status && c.status !== filter.status) return false;
    if (filter.authenticity === 'flagged' && !c.authenticity?.flagged) return false;
    if (filter.authenticity === 'ai_review' && !(c.civic_ai?.review_reasons?.length
      || ['registry_unconfirmed', 'unresolved'].includes(c.civic_ai?.department_basis))) return false;
    if (searchTerm && !c.ticket_id?.toLowerCase().includes(searchTerm.toLowerCase()) &&
        !c.description?.toLowerCase().includes(searchTerm.toLowerCase())) return false;
    return true;
  });

  if (authLoading || loading) return <LoadingSpinner />;
  if (!isAdmin) return null;

  const defaultCenter = [12.9716, 77.5946];

  const p1Count = stats?.by_priority?.P1 || 0;
  const p2Count = stats?.by_priority?.P2 || 0;

  const handleDeleteComplaint = async (complaintId) => {
    await api.delete(`/admin/complaints/${complaintId}`);
    setComplaints(prev => prev.filter(c => c.id !== complaintId));
    try {
      const statsData = await api.get('/admin/stats');
      setStats(statsData);
    } catch {}
  };

  return (
    <div className="container" style={{ paddingTop: 'var(--space-6)', paddingBottom: 'var(--space-12)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-6)' }}>
        <div>
          <h1 style={{ marginBottom: 'var(--space-1)' }}>Admin Dashboard</h1>
          <p className="text-secondary">Overview of civic complaints across Bengaluru</p>
        </div>
        <button className="btn btn-secondary" onClick={fetchDashboardData}>↻ Refresh</button>
      </div>

      {/* Stats Row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 'var(--space-4)', marginBottom: 'var(--space-8)' }}>
        <StatsCard icon="📝" value={stats?.total_complaints || 0} label="Total" />
        <StatsCard icon="⚡" value={stats?.total_open || 0} label="Open" />
        <StatsCard icon="✅" value={stats?.total_resolved || 0} label="Resolved" />
        <StatsCard icon="🔗" value={stats?.total_master_tickets || 0} label="Master Tickets" />
        <StatsCard icon="🚨" value={p1Count} label="P1 Critical" />
        <StatsCard icon="⚠️" value={p2Count} label="P2 High" />
      </div>

      {/* Map */}
      <div className="card" style={{ padding: 0, overflow: 'hidden', marginBottom: 'var(--space-8)' }}>
        <div style={{ height: '400px' }}>
          <MapContainer center={defaultCenter} zoom={12} style={{ height: '100%', width: '100%' }}>
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            {complaints.filter(c => c.location?.lat && c.location?.lon).map(c => (
              <Marker
                key={c.id}
                position={[c.location.lat, c.location.lon]}
                icon={createPriorityIcon(c.priority)}
              >
                <Popup>
                  <div style={{ minWidth: '180px' }}>
                    <strong style={{ fontFamily: 'var(--font-mono)', fontSize: '13px' }}>{c.ticket_id}</strong>
                    <div style={{ margin: '6px 0', fontSize: '14px' }}>{c.description?.slice(0, 80)}{c.description?.length > 80 ? '...' : ''}</div>
                    <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                      <span className={`badge badge-${c.priority?.toLowerCase()}`}>{c.priority}</span>
                      <span className="badge" style={{ background: '#F1F5F9', color: '#475569' }}>{c.department}</span>
                    </div>
                  </div>
                </Popup>
              </Marker>
            ))}
          </MapContainer>
        </div>
      </div>

      {/* Filters + Complaint List */}
      <div className="card">
        <h2 style={{ marginBottom: 'var(--space-4)' }}>All Complaints</h2>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 'var(--space-3)', marginBottom: 'var(--space-6)' }}>
          <input
            type="text"
            className="form-input"
            placeholder="Search by ticket ID or description..."
            value={searchTerm}
            onChange={e => setSearchTerm(e.target.value)}
            style={{ flex: '1', minWidth: '200px' }}
          />
          <select className="form-input" value={filter.dept} onChange={e => setFilter({...filter, dept: e.target.value})} style={{ width: 'auto', minWidth: '160px' }}>
            <option value="">All Departments</option>
            {[...new Set(complaints.map(c => c.department).filter(Boolean))].sort().map(d => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
          <select className="form-input" value={filter.priority} onChange={e => setFilter({...filter, priority: e.target.value})} style={{ width: 'auto', minWidth: '130px' }}>
            <option value="">All Priorities</option>
            <option value="P1">P1 Critical</option>
            <option value="P2">P2 High</option>
            <option value="P3">P3 Medium</option>
            <option value="P4">P4 Low</option>
            <option value="P5">P5 Minimal</option>
          </select>
          <select className="form-input" value={filter.status} onChange={e => setFilter({...filter, status: e.target.value})} style={{ width: 'auto', minWidth: '130px' }}>
            <option value="">All Statuses</option>
            <option value="submitted">Submitted</option>
            <option value="verified">Verified</option>
            <option value="in_progress">In Progress</option>
            <option value="resolved">Resolved</option>
          </select>
          <select className="form-input" value={filter.authenticity} onChange={e => setFilter({...filter, authenticity: e.target.value})} style={{ width: 'auto', minWidth: '150px' }}>
            <option value="">All Photos</option>
            <option value="flagged">Suspected Fake</option>
            <option value="ai_review">Needs AI review</option>
          </select>
        </div>

        {filteredComplaints.length > 0 ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 'var(--space-4)' }}>
            {filteredComplaints.map(c => (
              <ComplaintCard
                key={c.id}
                complaint={c}
                onClick={() => navigate(`/track/${c.ticket_id}`)}
                onDelete={handleDeleteComplaint}
              />
            ))}
          </div>
        ) : (
          <div style={{ textAlign: 'center', padding: 'var(--space-12)', color: 'var(--color-text-secondary)' }}>
            <p style={{ fontSize: '18px', marginBottom: 'var(--space-2)' }}>No complaints match your filters</p>
            <p>Try adjusting the filters above</p>
          </div>
        )}
      </div>
    </div>
  );
}
