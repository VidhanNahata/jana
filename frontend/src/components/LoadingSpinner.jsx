import React from 'react';

export function LoadingSpinner({ text }) {
  return (
    <div className="page-loader">
      <div className="spinner"></div>
      {text && <p>{text}</p>}
    </div>
  );
}
