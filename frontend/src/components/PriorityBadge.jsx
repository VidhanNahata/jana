import React from 'react';

export function PriorityBadge({ priority }) {
  if (!priority) return null;
  const pStr = priority.toString().toLowerCase();
  
  let pClass = 'badge-p5';
  if (pStr.includes('p1')) pClass = 'badge-p1';
  else if (pStr.includes('p2')) pClass = 'badge-p2';
  else if (pStr.includes('p3')) pClass = 'badge-p3';
  else if (pStr.includes('p4')) pClass = 'badge-p4';

  return (
    <span className={`badge ${pClass}`}>
      {priority}
    </span>
  );
}
