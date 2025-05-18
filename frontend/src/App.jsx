import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const BACKEND_URL = 'http://localhost:5001/api/check_typos'; // Flask URL
const TOP_K = 3;                                            // how many probs to show

function App() {
  const [inputText,      setInputText]      = useState('');
  const [originalText,   setOriginalText]   = useState('');
  const [correctedText,  setCorrectedText]  = useState('');
  const [tokenDetails,   setTokenDetails]   = useState([]);  // NEW
  const [metadata,       setMetadata]       = useState(null);
  const [isLoading,      setIsLoading]      = useState(false);
  const [error,          setError]          = useState('');

  const handleInputChange = (e) => setInputText(e.target.value);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputText.trim()) { setError('Please enter text.'); return; }

    setIsLoading(true); setError('');
    try {
      const res = await axios.post(BACKEND_URL, {
        sentence: inputText,
        top_k: TOP_K
      });
      const d = res.data;
      setOriginalText(d.original_sentence);
      setCorrectedText(d.corrected_sentence);
      setTokenDetails(d.token_details || []);
      setMetadata({
        modelName: d.model_name,
        processingTimeMs: d.processing_time_ms,
        correctionsMade: d.corrections_made,
        message: d.message
      });
    } catch (err) {
      setError(err.response?.data?.error || 'Server error'); }
    finally { setIsLoading(false); }
  };

  // helper to pretty-print top_probs dict
  const renderProbs = (obj) =>
    Object.entries(obj)
      .map(([tag,p]) => `${tag}: ${(p*100).toFixed(1)}%`)
      .join(' | ');

  return (
    <div className="app-container">
      <h1>Typo Detector AI</h1>

      <form onSubmit={handleSubmit}>
        <textarea
          value={inputText}
          onChange={handleInputChange}
          placeholder="Enter a sentence with potential typos…"
          rows={4}
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading}>
          {isLoading ? 'Checking…' : 'Check Typos'}
        </button>
      </form>

      {isLoading && <p>Loading…</p>}
      {error     && <p className="error-message">{error}</p>}

      {correctedText && (
        <div className="results-container">
          <h3>Original Sentence</h3><p>{originalText}</p>
          <h3>Corrected Sentence</h3><p>{correctedText}</p>

          {/* ---------- per-token table ---------- */}
          {tokenDetails.length > 0 && (
            <>
              <h3>Token-level details (top {TOP_K})</h3>
              <table className="token-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Token</th>
                    <th>Predicted Tag</th>
                    <th>Top Probabilities</th>
                  </tr>
                </thead>
                <tbody>
                  {tokenDetails.map((t, idx) => (
                    <tr key={idx}>
                      <td>{idx+1}</td>
                      <td>{t.token}</td>
                      <td>{t.pred_tag}</td>
                      <td>{renderProbs(t.top_probs)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}

          {metadata && (
            <div className="metadata">
              <p><strong>Corrections:</strong> {metadata.correctionsMade ? 'Yes' : 'No'}</p>
              <p><strong>Model:</strong> {metadata.modelName}</p>
              <p><strong>Time:</strong> {metadata.processingTimeMs} ms</p>
              <p><strong>Message:</strong> {metadata.message}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
