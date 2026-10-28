import React from 'react';

// Review step for the civic AI analysis: the citizen checks what the photo shows,
// picks the primary issue and the responsible authority, and edits the draft
// before anything is submitted. Every choice re-runs classification and routing
// on the server (cheap: the image is not analysed again).

const BLR_CORPS = ['KA.BLR.CENTRAL', 'KA.BLR.NORTH', 'KA.BLR.SOUTH', 'KA.BLR.EAST', 'KA.BLR.WEST'];

const STATUS_TEXT = {
  ok: null,
  needs_citizen_input: 'Please answer the questions below before submitting.',
  needs_review: 'You can submit; an official will double-check the details marked below.',
  out_of_scope: null,
  error: 'The analysis did not complete. You can still submit; officials will classify it.',
};

const CONSISTENCY_WARNING = {
  contradiction: 'The photo seems to show a different problem from the one you describe.',
  unrelated_image: 'The photo does not look like a street or public place.',
  insufficient_image_evidence: 'The problem you describe is not clearly visible in the photo.',
};

function pct(x) {
  return `${Math.round((x || 0) * 100)}%`;
}

export function AiReview({ analysis, taxonomy, choices, onChoose, draftText, setDraftText, busy }) {
  const { classification: cls, routing, image_analysis: img, fusion, complaint_draft: draft } = analysis;
  const name = (id) => taxonomy[id]?.name_en || id;
  const outOfScope = analysis.status === 'out_of_scope';

  const issueOptions = [cls.subcategory_id, ...cls.alternatives.map(a => a.subcategory_id)]
    .filter((id, i, arr) => id && !id.startsWith('OUT_OF_SCOPE') && arr.indexOf(id) === i);
  const selectedIssue = choices.confirmed_subcategory_id || cls.subcategory_id;

  const authorityOptions = [
    ...(routing.authority_id ? [{ authority_id: routing.authority_id, authority_name: routing.authority_name, reason: 'Suggested' }] : []),
    ...routing.alternatives,
  ].filter((a, i, arr) => arr.findIndex(b => b.authority_id === a.authority_id) === i);
  const needsCorporation = routing.missing_location_fields.includes('city_corporation');
  const selectedAuthority = choices.confirmed_authority_id || choices.city_corporation_id
    || (needsCorporation ? null : routing.authority_id);

  const pickAuthority = (id) => {
    // A Bengaluru corporation is a location answer: re-route with it
    if (BLR_CORPS.includes(id)) onChoose({ city_corporation_id: id, confirmed_authority_id: null });
    else onChoose({ confirmed_authority_id: id }, { rerun: false });
  };

  return (
    <div className="ai-review mb-6">
      <h3 className="mb-2">Check before submitting</h3>

      {outOfScope ? (
        <div className="authenticity-banner mb-4">
          <strong>This doesn't look like a civic infrastructure problem</strong>
          <div className="text-sm">{cls.out_of_scope_reason}</div>
          <div className="text-sm text-secondary mt-1">
            This platform handles roads, drains, water, waste, streetlights and other public infrastructure.
            If that is wrong, choose the problem type below.
          </div>
        </div>
      ) : STATUS_TEXT[analysis.status] && (
        <div className="detection-banner mb-4 text-sm">{STATUS_TEXT[analysis.status]}</div>
      )}

      {CONSISTENCY_WARNING[fusion.consistency] && (
        <div className="authenticity-banner mb-4 text-sm">
          <strong>⚠️ {CONSISTENCY_WARNING[fusion.consistency]}</strong>
          <div className="text-secondary">Consider adding a clearer photo of the problem.</div>
        </div>
      )}

      {analysis.authenticity?.flagged && (
        <div className="authenticity-banner mb-4 text-sm">
          <strong>⚠️ Photo check: {analysis.authenticity.reasons.join('; ')}</strong>
          <div className="text-secondary">The report will be marked for verification.</div>
        </div>
      )}

      <div className="ai-evidence mb-4">
        <div>
          <div className="ai-label">What the photo shows (automated)</div>
          {img.status === 'ok' ? (
            <>
              {img.scene_description && <p className="text-sm">{img.scene_description}</p>}
              {img.problems.length ? (
                <ul className="text-sm">
                  {img.problems.map((p, i) => (
                    <li key={i}>{name(p.subcategory_id)} — {p.evidence} ({pct(p.confidence)})</li>
                  ))}
                </ul>
              ) : <p className="text-sm text-secondary">No listed problem clearly visible.</p>}
            </>
          ) : (
            <p className="text-sm text-secondary">
              {img.problems.length
                ? img.problems.map(p => p.evidence).join('; ')
                : img.status === 'no_input' ? 'No photo.' : 'Photo understanding is not available on this server.'}
            </p>
          )}
        </div>
        <div>
          <div className="ai-label">What you reported</div>
          <p className="text-sm">“{analysis.text_analysis.original_text || '—'}”</p>
        </div>
      </div>

      <div className="form-group mb-4">
        <label className="form-label">
          {cls.primary_selection_required ? 'Several problems found — which is the main one?' : 'Problem type'}
        </label>
        {issueOptions.map(id => (
          <label key={id} className="ai-option">
            <input
              type="radio"
              name="issue"
              checked={!outOfScope && selectedIssue === id && !(cls.primary_selection_required && !choices.confirmed_subcategory_id)}
              onChange={() => onChoose({ confirmed_subcategory_id: id })}
              disabled={busy}
            />
            <span>{name(id)}</span>
            {id === cls.subcategory_id && !outOfScope && <span className="text-sm text-tertiary"> · {pct(cls.confidence)}</span>}
          </label>
        ))}
        <select
          className="form-select mt-1"
          value=""
          onChange={(e) => e.target.value && onChoose({ confirmed_subcategory_id: e.target.value })}
          disabled={busy}
        >
          <option value="">Something else…</option>
          {Object.entries(taxonomy)
            .filter(([id]) => !id.startsWith('OUT_OF_SCOPE'))
            .map(([id, s]) => <option key={id} value={id}>{s.category_name_en} — {s.name_en}</option>)}
        </select>
      </div>

      {!outOfScope && (
        <div className="form-group mb-4">
          <label className="form-label">Send to</label>
          {routing.status === 'location_required' && (
            <p className="text-sm form-error">Add the locality, city or PIN code above and press “Re-check”.</p>
          )}
          {needsCorporation && (
            <p className="text-sm text-secondary">
              Bengaluru is served by five city corporations. Choose the one for this locality
              (shown on your property tax receipt or at gba.karnataka.gov.in).
            </p>
          )}
          {authorityOptions.map(a => (
            <label key={a.authority_id} className="ai-option">
              <input
                type="radio"
                name="authority"
                checked={selectedAuthority === a.authority_id}
                onChange={() => pickAuthority(a.authority_id)}
                disabled={busy}
              />
              <span>{a.authority_name}</span>
              {a.reason && a.reason !== 'Suggested' && <span className="text-sm text-tertiary"> · {a.reason}</span>}
            </label>
          ))}
          {routing.verified_url && (
            <div className="text-sm text-secondary mt-1">
              Official site: <a href={routing.verified_url} target="_blank" rel="noreferrer">{routing.verified_url}</a>
            </div>
          )}
          {routing.routing_rationale.length > 0 && (
            <details className="text-sm text-secondary mt-1">
              <summary>Why this authority?</summary>
              <ul>{routing.routing_rationale.map((r, i) => <li key={i}>{r}</li>)}</ul>
            </details>
          )}
        </div>
      )}

      {draft && (
        <div className="form-group mb-4">
          <div className="flex justify-between items-center">
            <label className="form-label">Complaint text (you can edit it)</label>
            <div className="ai-lang">
              {['en', 'kn'].map(l => (
                <button
                  key={l}
                  type="button"
                  className={`btn btn-sm ${draft.language === l ? 'btn-primary' : 'btn-outline'}`}
                  onClick={() => onChoose({ draft_language: l })}
                  disabled={busy || draft.language === l}
                >
                  {l === 'en' ? 'English' : 'ಕನ್ನಡ'}
                </button>
              ))}
            </div>
          </div>
          <textarea
            className="form-input ai-draft"
            rows={12}
            value={draftText}
            onChange={(e) => setDraftText(e.target.value)}
          />
          {draft.missing_details.length > 0 && (
            <div className="text-sm text-secondary mt-1">Missing: {draft.missing_details.join(', ')}</div>
          )}
        </div>
      )}

      {cls.review_reasons.length > 0 && !outOfScope && (
        <details className="text-sm text-secondary">
          <summary>Notes for the reviewing official</summary>
          <ul>{cls.review_reasons.map((r, i) => <li key={i}>{r}</li>)}</ul>
        </details>
      )}
    </div>
  );
}

export function canSubmit(analysis, choices) {
  if (!analysis) return true;
  if (analysis.status === 'out_of_scope') return false;
  const { classification: cls, routing } = analysis;
  if (cls.primary_selection_required && !choices.confirmed_subcategory_id) return false;
  if (routing.missing_location_fields.includes('city_corporation') && !choices.confirmed_authority_id) return false;
  return true;
}
