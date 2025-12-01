# 🔬 Breast Cancer Histopathology Analysis

AI-powered breast cancer detection from histopathology images using Deep Learning.

## Features

- 🖼️ **Image Analysis**: Upload histopathology images for classification
- 🤖 **Deep Learning**: EfficientNet-B0 with Coordinate Attention mechanism
- 📊 **Confidence Scores**: Get probability distribution for each class
- 📜 **History**: Track all past predictions
- 🔗 **REST API**: Full API access for integration
- ☁️ **Cloud Storage**: Images stored in Vercel Blob Storage
- 💾 **Database**: MongoDB for prediction history

## Project Structure

```
deployment/
├── main.py              # FastAPI + Gradio entry point
├── gradio_app.py        # Gradio UI components
├── core/
│   ├── __init__.py      # Module exports
│   ├── model.py         # ML model handler
│   ├── database.py      # MongoDB handler
│   └── storage.py       # Vercel Blob handler
├── best_histopathology_model.pth  # Trained model weights
├── requirements.txt     # Python dependencies
├── Procfile            # Render deployment config
├── .env.example        # Environment variables template
└── uploads/            # Local file storage fallback
```

## Quick Start

### 1. Clone & Setup

```bash
cd deployment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

### 3. Run Locally

```bash
python main.py
```

Open http://localhost:8000

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `MONGODB_URI` | MongoDB connection string | For history |
| `BLOB_READ_WRITE_TOKEN` | Vercel Blob token | For cloud storage |
| `PORT` | Server port (default: 8000) | No |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/stats` | Database statistics |
| POST | `/api/predict` | Analyze image |
| GET | `/api/history` | Get recent predictions |
| GET | `/api/prediction/{id}` | Get prediction details |
| DELETE | `/api/prediction/{id}` | Delete prediction |

### Example: Analyze Image

```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.png"
```

Response:
```json
{
  "id": "507f1f77bcf86cd799439011",
  "filename": "image.png",
  "prediction": "Benign",
  "confidence": 0.95,
  "benign_probability": 0.95,
  "malignant_probability": 0.05,
  "image_url": "https://..."
}
```

## Deploy to Render

### 1. Create Render Web Service

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Root Directory**: `deployment`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python main.py`
   - **Instance Type**: Starter ($7/month) or higher

### 2. Add Environment Variables

In Render dashboard → Environment:

```
MONGODB_URI=mongodb+srv://...
BLOB_READ_WRITE_TOKEN=vercel_blob_rw_...
```

### 3. Deploy

Render will automatically deploy on push to main branch.

## MongoDB Setup (Free)

1. Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Create free cluster (M0 tier)
3. Create database user
4. Get connection string
5. Add to environment variables

## Vercel Blob Setup

1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Create or select a project
3. Go to Storage → Create → Blob Store
4. Create token with read/write permissions
5. Add to environment variables

## Model Details

- **Architecture**: EfficientNet-B0 + Coordinate Attention
- **Input Size**: 160x160 RGB
- **Classes**: Benign, Malignant
- **Output**: Softmax probabilities

## License

MIT License - For research and educational purposes only.

## Disclaimer

⚠️ This tool is for research and educational purposes only. It is NOT a substitute for professional medical diagnosis. Always consult qualified medical professionals for actual diagnosis and treatment decisions.
