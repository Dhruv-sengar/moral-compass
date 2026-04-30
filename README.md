# 🧭 Moral Compass Classifier

A full-stack machine learning project that classifies any moral scenario into one of three categories:

| Category | Description | AITA Mapping |
|---|---|---|
| **Utilitarian** | Maximises overall good for the greatest number | Custom synthetic |
| **Ethical** | Principled, fair, honest behaviour | NTA (Not The Asshole) |
| **Selfish** | Self-centred / harmful to others | YTA (You're The Asshole) |

---

## 🗂️ Project Structure

```
MORAL COMPASS/
├── backend/
│   └── main.py            ← FastAPI REST API (/predict, /health, /classes)
├── src/
│   ├── data_generation.py ← Synthetic dataset (English + Hinglish scenarios)
│   ├── train.py           ← ML pipeline (TF-IDF + LR + SVM)
│   └── predict.py         ← Prediction + explainability helpers
├── data/
│   └── moral_dataset.csv  ← Training data (auto-generated)
├── models/
│   ├── best_model.pkl     ← Saved best model (after training)
│   └── vectorizer.pkl     ← Saved TF-IDF vectoriser
├── frontend/
│   ├── src/
│   │   ├── App.jsx        ← Main React component
│   │   ├── index.css      ← Dark-mode design system
│   │   └── main.jsx       ← React entry point
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+
- Node.js 18+
- npm 9+

---

### 1. Clone & create virtual environment

```bash
cd "D:\PROJECTS\MORAL COMPASS"
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Generate dataset (if not already present)

```bash
python src/data_generation.py
```

This creates `data/moral_dataset.csv` with ~300 labelled moral scenarios (English + Hinglish).

---

### 4. (Optional) Merge Kaggle AITA dataset

Download the Reddit AITA dataset from Kaggle:

```bash
pip install kaggle
# Put your kaggle.json in ~/.kaggle/
kaggle datasets download -d kreative/amitheasshole-reddit-posts-and-comments
unzip amitheasshole-reddit-posts-and-comments.zip -d data/raw/
```

Then run the merge script (add your own merge logic in `src/data_generation.py`).

---

### 5. Train the models

```bash
python src/train.py
```

This will:
- Load `data/moral_dataset.csv`
- Fit a TF-IDF vectoriser (unigrams + bigrams, 5 000 features)
- Train **Logistic Regression** and **linear SVM**
- Print accuracy, classification report, confusion matrix & cross-validation scores
- Save the best model to `models/best_model.pkl` and `models/vectorizer.pkl`

---

### 6. Start the FastAPI backend

```bash
uvicorn backend.main:app --reload --port 8000
```

API will be available at **http://localhost:8000**
Interactive docs: **http://localhost:8000/docs**

---

### 7. Start the React frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend will be available at **http://localhost:5173**

---

## 🔌 API Reference

### `POST /predict`

Classify a moral scenario.

**Request**
```json
{ "text": "I donated my kidney to a stranger to save their life." }
```

**Response**
```json
{
  "prediction":     "Utilitarian",
  "confidence":     0.9123,
  "confidence_pct": "91.2%",
  "probabilities": {
    "Ethical":     0.0421,
    "Selfish":     0.0456,
    "Utilitarian": 0.9123
  },
  "top_features": [
    { "word": "donated kidney", "score": 0.3241 },
    { "word": "stranger save",  "score": 0.2180 },
    { "word": "life",           "score": 0.1502 }
  ]
}
```

### `GET /health`
Returns `{ "status": "ok", "model_loaded": true/false }`

### `GET /classes`
Returns `{ "classes": ["Ethical", "Selfish", "Utilitarian"] }`

---

## 🧪 Example cURL

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Maine apne intern ka kaam chura liya aur boss ko apna bataya."}'
```

---

## 🎨 Frontend Features

- **Dark glassmorphism UI** with animated background grid
- **Quick-fill sample scenarios** (English + Hinglish)
- **Confidence progress bar** with colour-coded categories
- **Per-class probability bars** (Utilitarian / Ethical / Selfish)
- **Explainability panel** — keywords that drove the prediction
- **Keyboard shortcut**: `Ctrl+Enter` to analyze
- **Responsive** — works on mobile and desktop

---

## 🚀 Deployment

### Backend → Render / Railway

1. Push to GitHub
2. Create a new **Web Service** on [Render](https://render.com)
3. Set build command: `pip install -r requirements.txt && python src/train.py`
4. Set start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

### Frontend → Vercel / Netlify

1. `cd frontend && npm run build`
2. Deploy the `frontend/dist/` folder
3. Set environment variable: `VITE_API_URL=https://your-backend.onrender.com`

---

## 📊 Model Performance (typical on synthetic dataset)

| Metric | Logistic Regression | SVM (linear) |
|--------|--------------------:|-------------:|
| Accuracy | ~92–95% | ~91–94% |
| Weighted F1 | ~0.92 | ~0.91 |

*Exact numbers depend on random seed and dataset size.*

---

## 🌐 Bonus Features

- ✅ **Hinglish scenarios** — 20+ Indian moral dilemmas per class
- ✅ **"Why this prediction?"** — TF-IDF contribution scores
- ✅ **Dark mode UI** — glassmorphism, animated gradients
- ✅ **Per-class probability chart** — not just the top prediction
- ✅ **Keyboard shortcut** — Ctrl/Cmd + Enter

---

*Built with Python · scikit-learn · FastAPI · React · Vite*
