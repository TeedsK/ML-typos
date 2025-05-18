# Typo Detector AI

End-to-end spell-and-typo fixer tailored to software-engineering text.  
*Edit-tag* architecture (RoBERTa) → Flask API → React Front-end.

---

## Features

| Layer | What it does |
|-------|--------------|
| **Model** | Fine-tuned *edit-tag* RoBERTa (60 k curated pairs) <br>• outputs corrected sentence <br>• per-token tag + top-*k* probability distribution |
| **Backend (Flask)** | `/api/check_typos` POST endpoint <br>• JSON in → JSON out <br>• CORS-ready for React |
| **Frontend (React)** | Text box, 1-click correction, table of token-level probabilities |

---

## Directory layout
```
├── README.md ← (this file)
├── frontend/ ← React app
│ └── src/App.jsx
├── backend/
│ ├── app.py ← Flask server
│ ├── model_loader.py ← bridges model ⇆ Flask
│ └── README.md ← backend-specific docs
└── edit_tag_spellfix/ ← Python package for training + inference
├── tags.py ├── model.py
├── data.py ├── train.py
└── predict_verbose.py
```


---

## Quick start

```bash
# 1. clone repo & enter
git clone https://github.com/your-org/typo-detector-ai.git
cd typo-detector-ai

# 2. backend
cd backend
python -m venv .venv && . .venv/bin/activate
pip install -r ../requirements.txt
python app.py            # starts on http://localhost:5001
# leave running in its tab

# 3. frontend
cd ../frontend
npm install
npm start                # opens http://localhost:3000
```

## API reference

POST /api/check_typos

```
Request body
{
  "sentence": "a machine learning and fuil stack engineer ...",
  "top_k": 3                // optional, default = 3
}

Success 200
{
  "original_sentence":  "...",
  "corrected_sentence": "...",
  "token_details": [        // one entry per original token
    {
      "token": "fuil",
      "pred_tag": "REPLACE_full-stack",
      "top_probs": {
        "REPLACE_full-stack": 0.97,
        "KEEP":                0.02,
        "DELETE":              0.01
      }
    },
    ...
  ],
  "model_name": "edit-tag-roberta-60k-v2",
  "processing_time_ms": 12.8,
  "corrections_made": true,
  "message": "Typos checked successfully."
}
```

## Future Roadmap

| Stage                   | Goal                                                                   | Notes                                    |
| ----------------------- | ---------------------------------------------------------------------- | ---------------------------------------- |
| **Data v3**             | Hard-negative mining loop – feed the model its own remaining mistakes. | ≈ +0.5 pp accuracy, <30 min work.        |
| **Gap-head**            | Add second classifier for insertions.                                  | Fixes dropped words (articles, hyphens). |
| **Encoder tuning**      | Unfreeze top 4 layers for a single epoch.                              | Improves subtle grammar / casing.        |
| **Character-noise aug** | Keyboard-distance, phonetic swaps, Unicode twins.                      | Better coverage of unseen typos.         |
| **ONNX export**         | 2-3 × faster CPU inference, <100 MB model.                             | Production deployment.                   |
| **CI regression suite** | 200-line golden file, blocks perf regressions.                         | Keeps quality steady over time.          |
