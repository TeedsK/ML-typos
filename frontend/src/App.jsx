import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css'; // Assuming you might add specific styles here if needed

const BACKEND_URL = 'http://localhost:5001/api/check_typos'; // Backend API URL

function App() {
  const [inputText, setInputText] = useState('');
  const [originalText, setOriginalText] = useState('');
  const [correctedText, setCorrectedText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [metadata, setMetadata] = useState(null); // For model name, processing time etc.

  useEffect(() => {
    console.log('[App.jsx] Component mounted.');
    // You could add an initial health check to the backend here if desired
    // checkBackendHealth();
    return () => {
      console.log('[App.jsx] Component unmounted.');
    };
  }, []);

  // Example of a health check function (optional)
  // const checkBackendHealth = async () => {
  //   try {
  //     const response = await axios.get('http://localhost:5001/api/health');
  //     console.log('[App.jsx checkBackendHealth] Backend health:', response.data);
  //   } catch (err) {
  //     console.error('[App.jsx checkBackendHealth] Backend health check failed:', err);
  //     setError('Failed to connect to the backend. Ensure it is running.');
  //   }
  // };

  const handleInputChange = (event) => {
    console.debug('[App.jsx handleInputChange] New input value:', event.target.value);
    setInputText(event.target.value);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    console.info('[App.jsx handleSubmit] Submitting sentence:', inputText);
    if (!inputText.trim()) {
      console.warn('[App.jsx handleSubmit] Input text is empty. Aborting submission.');
      setError('Please enter some text to check.');
      setOriginalText('');
      setCorrectedText('');
      setMetadata(null);
      return;
    }

    setIsLoading(true);
    setError('');
    setOriginalText(''); // Clear previous results
    setCorrectedText('');
    setMetadata(null);

    try {
      console.debug('[App.jsx handleSubmit] Sending POST request to:', BACKEND_URL);
      const response = await axios.post(BACKEND_URL, { sentence: inputText });
      console.info('[App.jsx handleSubmit] Response received:', response);

      if (response.data) {
        console.debug('[App.jsx handleSubmit] Response data:', response.data);
        setOriginalText(response.data.original_sentence);
        setCorrectedText(response.data.corrected_sentence);
        setMetadata({
          modelName: response.data.model_name,
          processingTimeMs: response.data.processing_time_ms,
          correctionsMade: response.data.corrections_made,
          message: response.data.message
        });
        console.log('[App.jsx handleSubmit] State updated with API response.');
      } else {
        console.error('[App.jsx handleSubmit] Received empty data in response.');
        setError('Received an empty response from the server.');
      }
    } catch (err) {
      console.error('[App.jsx handleSubmit] API Error:', err);
      let errorMessage = 'An error occurred while checking typos.';
      if (err.response) {
        // Server responded with a status code outside the 2xx range
        console.error('[App.jsx handleSubmit] Error response data:', err.response.data);
        console.error('[App.jsx handleSubmit] Error response status:', err.response.status);
        errorMessage = err.response.data?.error || err.response.data?.message || `Server error: ${err.response.status}`;
      } else if (err.request) {
        // Request was made but no response received
        console.error('[App.jsx handleSubmit] Error request:', err.request);
        errorMessage = 'No response from server. Please check if the backend is running.';
      } else {
        // Something else happened in setting up the request
        console.error('[App.jsx handleSubmit] Error message:', err.message);
        errorMessage = `Error: ${err.message}`;
      }
      setError(errorMessage);
    } finally {
      console.debug('[App.jsx handleSubmit] Setting isLoading to false.');
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <h1>Typo Detector AI</h1>
      <form onSubmit={handleSubmit}>
        <textarea
          value={inputText}
          onChange={handleInputChange}
          placeholder="Enter a sentence with potential typos..."
          rows="4"
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading}>
          {isLoading ? 'Checking...' : 'Check Typos'}
        </button>
      </form>

      {isLoading && <p className="loading-message">Loading results...</p>}
      {error && <p className="error-message">Error: {error}</p>}

      {correctedText && (
        <div className="results-container">
          <h3>Original Sentence:</h3>
          <p>{originalText}</p>
          <h3>Corrected Sentence:</h3>
          <p>{correctedText}</p>
          {metadata && (
            <div className="metadata">
              <p>
                <strong>Corrections made:</strong> {metadata.correctionsMade ? 'Yes' : 'No'} <br />
                <strong>Model:</strong> {metadata.modelName} <br />
                <strong>Processing Time:</strong> {metadata.processingTimeMs} ms <br />
                <strong>Server Message:</strong> {metadata.message}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;