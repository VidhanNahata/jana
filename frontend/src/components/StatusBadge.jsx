import React from 'react';

export function StatusBadge({ status }) {
  if (!status) return null;
  const sStr = status.toString().toLowerCase();
  
  let sClass = 'badge-submitted';
  if (sStr.includes('verified')) sClass = 'badge-verified';
  else if (sStr.includes('progress')) sClass = 'badge-inprogress';
  else if (sStr.includes('resolved') || sStr.includes('closed')) sClass = 'badge-resolved';
  else if (sStr.includes('rejected')) sClass = 'badge-rejected';

  return (
    <span className={`badge ${sClass}`}>
      {status}
    </span>
  );
}
