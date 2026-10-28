import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { api } from '../utils/api';
import { ComplaintCard } from '../components/ComplaintCard';
import { LoadingSpinner } from '../components/LoadingSpinner';

export function MyComplaintsPage() {
  const { isLoggedIn, loading: authLoading } = useAuth();
  const navigate = useNavigate();
  const [complaints, setComplaints] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!authLoading && !isLoggedIn) {
      navigate('/login');
    } else if (isLoggedIn) {
      fetchComplaints();
    }
  }, [isLoggedIn, authLoading, navigate]);

  const fetchComplaints = async () => {
    try {
      const data = await api.get('/complaints/my');
      setComplaints(data);
    } catch (err) {
      setError('Failed to load complaints.');
    } finally {
      setLoading(false);
    }
  };

  if (authLoading || loading) return <LoadingSpinner text="Loading your complaints..." />;

  return (
    <div className="container" style={{ paddingTop: 'var(--space-8)' }}>
      <div className="mb-8">
        <h1>My Complaints</h1>
        <p className="text-secondary">Track issues you've reported</p>
      </div>

      {error && <div className="form-error mb-6">{error}</div>}

      {!loading && complaints.length === 0 ? (
        <div className="card text-center" style={{ padding: 'var(--space-12)' }}>
          <div style={{ fontSize: '48px', marginBottom: '16px' }}>📝</div>
          <h3 className="mb-2">No complaints yet</h3>
          <p className="text-secondary mb-6">You haven't reported any issues. Help improve the city by reporting problems you see.</p>
          <button className="btn btn-primary" onClick={() => navigate('/')}>Report an Issue</button>
        </div>
      ) : (
        <div className="grid grid-cols-3 gap-6">
          {complaints.map(complaint => (
            <ComplaintCard 
              key={complaint.id} 
              complaint={complaint} 
              onClick={() => navigate(`/track/${complaint.ticket_id}`)}
            />
          ))}
        </div>
      )}
    </div>
  );
}
