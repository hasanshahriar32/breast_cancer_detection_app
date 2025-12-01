"""
Breast Cancer Histopathology Analysis - Main Application
Single-port deployment for Render: FastAPI + Gradio on same port.
"""

import os
import uvicorn
import gradio as gr
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.convertors import Convertor, CONVERTOR_TYPES
from pydantic import BaseModel

# Register custom path convertor for MongoDB ObjectIDs
class ObjectIDConvertor(Convertor):
    regex = "[0-9a-fA-F]{24}"

    def convert(self, value: str) -> str:
        return value

    def to_string(self, value: str) -> str:
        return value

CONVERTOR_TYPES["objectid"] = ObjectIDConvertor()
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import handlers
from core import ModelHandler, DatabaseHandler, StorageHandler
import gradio_app
import gradio_history

# Configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_histopathology_model.pth")
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Initialize handlers
model_handler = ModelHandler(MODEL_PATH)
db_handler = DatabaseHandler()
storage_handler = StorageHandler()

# Share handlers with gradio_app
gradio_app.model_handler = model_handler
gradio_app.db_handler = db_handler
gradio_app.storage_handler = storage_handler

# Share handlers with gradio_history
gradio_history.model_handler = model_handler
gradio_history.db_handler = db_handler
gradio_history.storage_handler = storage_handler


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Breast Cancer Histopathology Analysis API",
    description="""
    API for analyzing breast cancer histopathology images using Deep Learning.
    
    ## Features
    - 🔬 **Predict**: Analyze histopathology images for benign/malignant classification
    - 📜 **History**: View past predictions
    - 📋 **Details**: Get detailed results by prediction ID
    
    ## Model
    EfficientNet-B0 with Coordinate Attention mechanism trained on breast cancer histopathology dataset.
    """,
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded files
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")


# ============================================================================
# API Models
# ============================================================================

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    id: Optional[str] = None
    filename: str
    prediction: str
    confidence: float
    benign_probability: float
    malignant_probability: float
    image_url: Optional[str] = None
    attention_metrics: Optional[Dict[str, float]] = None


class PredictionDetail(BaseModel):
    """Detailed prediction model."""
    id: str
    filename: str
    prediction: str
    confidence: float
    benign_probability: float
    malignant_probability: float
    image_url: Optional[str] = None
    attention_metrics: Optional[Dict[str, float]] = None
    timestamp: Optional[str] = None
    created_at: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str
    model_loaded: bool
    database_connected: bool
    storage_configured: bool


class StatsResponse(BaseModel):
    """Statistics response."""
    database: Dict[str, Any]


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and component status."""
    return {
        "status": "ok",
        "message": "Breast Cancer Analysis API is running",
        "model_loaded": model_handler.model is not None,
        "database_connected": db_handler.is_connected,
        "storage_configured": storage_handler.is_configured
    }


@app.get("/api/stats", response_model=StatsResponse, tags=["Statistics"])
async def get_stats():
    """Get database statistics."""
    return {
        "database": db_handler.get_stats()
    }


@app.post("/api/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(..., description="Histopathology image file")):
    """
    Analyze a histopathology image for breast cancer classification.
    
    - **file**: Image file (PNG, JPG, etc.)
    
    Returns prediction with confidence scores.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        
        # Get prediction
        result = model_handler.predict(contents)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Upload to storage
        image_url = storage_handler.upload_file(contents, file.filename or "upload.png")
        
        # Save to database
        prediction_id = db_handler.save_prediction(image_url, result, file.filename or "upload.png")
        
        return {
            "id": prediction_id,
            "filename": file.filename or "upload.png",
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "benign_probability": result["benign_probability"],
            "malignant_probability": result["malignant_probability"],
            "image_url": image_url,
            "attention_metrics": result.get("attention_metrics")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history", response_model=List[Dict[str, Any]], tags=["History"])
async def get_history(
    limit: int = Query(default=10, ge=1, le=100, description="Maximum number of results")
):
    """
    Get recent prediction history.
    
    - **limit**: Maximum number of results (1-100, default 10)
    """
    return db_handler.get_recent_predictions(limit)


@app.get("/api/prediction/{prediction_id}", response_model=PredictionDetail, tags=["Prediction"])
async def get_prediction(prediction_id: str):
    """
    Get a specific prediction by ID.
    
    - **prediction_id**: MongoDB document ID
    """
    result = db_handler.get_prediction(prediction_id)
    
    if not result:
        raise HTTPException(status_code=404, detail=f"Prediction not found: {prediction_id}")
    
    return {
        "id": result.get("_id"),
        "filename": result.get("filename"),
        "prediction": result.get("prediction"),
        "confidence": result.get("confidence"),
        "benign_probability": result.get("benign_probability"),
        "malignant_probability": result.get("malignant_probability"),
        "image_url": result.get("image_url"),
        "attention_metrics": result.get("attention_metrics"),
        "timestamp": result.get("timestamp"),
        "created_at": result.get("created_at")
    }


@app.delete("/api/prediction/{prediction_id}", tags=["Prediction"])
async def delete_prediction(prediction_id: str):
    """
    Delete a prediction by ID.
    
    - **prediction_id**: MongoDB document ID
    """
    if not db_handler.is_connected:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    deleted = db_handler.delete_prediction(prediction_id)
    
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Prediction not found: {prediction_id}")
    
    return {"message": f"Prediction {prediction_id} deleted successfully"}


@app.get("/{prediction_id:objectid}", include_in_schema=False)
async def redirect_to_detail(prediction_id: str):
    """Redirect valid IDs to the history detail view."""
    return RedirectResponse(url=f"/history/?id={prediction_id}")


@app.get("/history/{prediction_id:objectid}", include_in_schema=False)
async def redirect_history_subpath(prediction_id: str):
    """Redirect /history/ID to /history?id=ID."""
    return RedirectResponse(url=f"/history/?id={prediction_id}")


@app.get("/history", include_in_schema=False)
async def redirect_history_root():
    """Redirect /history to /history/ to ensure Gradio app loads."""
    return RedirectResponse(url="/history/")

# ============================================================================
# Mount Gradio Apps
# ============================================================================

# Mount History/Detail Gradio at /history path
app = gr.mount_gradio_app(app, gradio_history.history_gradio_app, path="/history")

# Mount Main Gradio at root path
app = gr.mount_gradio_app(app, gradio_app.gradio_app, path="/")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    
    print("\n" + "=" * 60)
    print("🔬 Breast Cancer Histopathology Analysis")
    print("=" * 60)
    print(f"📍 Server: http://localhost:{port}")
    print(f"🖥️  Main UI: http://localhost:{port}")
    print(f"📜 History: http://localhost:{port}/history")
    print(f"📡 API Docs: http://localhost:{port}/api/docs")
    print(f"📖 ReDoc: http://localhost:{port}/api/redoc")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
