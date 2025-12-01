"""
Breast Cancer Histopathology Analysis - Gradio UI
Single-port deployment with FastAPI backend.
"""

import gradio as gr
from PIL import Image
import io
import os
from typing import Optional, Dict, Tuple, Any

from core import ModelHandler, DatabaseHandler, StorageHandler

# Initialize handlers (shared with main.py when imported)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_histopathology_model.pth")
model_handler: Optional[ModelHandler] = None
db_handler: Optional[DatabaseHandler] = None
storage_handler: Optional[StorageHandler] = None


def init_handlers():
    """Initialize handlers (called from main.py)."""
    global model_handler, db_handler, storage_handler
    model_handler = ModelHandler(MODEL_PATH)
    db_handler = DatabaseHandler()
    storage_handler = StorageHandler()


def analyze_image(image: Image.Image) -> Tuple[Optional[Dict], str]:
    """
    Analyze a histopathology image for cancer classification.
    
    Args:
        image: PIL Image to analyze
        
    Returns:
        Tuple of (probability dict, markdown report)
    """
    if image is None:
        return None, "⚠️ Please upload an image first."
    
    if model_handler is None:
        return None, "⚠️ Model not initialized."
    
    try:
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        
        # Get prediction
        result = model_handler.predict(img_bytes)
        
        if "error" in result:
            return None, f"❌ Error: {result['error']}"
        
        # Upload to storage
        filename = f"analysis_{os.urandom(4).hex()}.png"
        image_url = storage_handler.upload_file(img_bytes, filename) if storage_handler else None
        
        # Save to database
        prediction_id = None
        if db_handler and image_url:
            prediction_id = db_handler.save_prediction(image_url, result, filename)
        
        # Format results
        prediction = result['prediction']
        confidence = result['confidence'] * 100
        benign_prob = result['benign_probability'] * 100
        malignant_prob = result['malignant_probability'] * 100
        
        # Build report
        report = f"""
## 🔬 Analysis Results

### Prediction: **{prediction}**
### Confidence: **{confidence:.1f}%**

---

| Class | Probability |
|-------|-------------|
| Benign | {benign_prob:.1f}% |
| Malignant | {malignant_prob:.1f}% |

---

### 💡 Clinical Insights
"""
        
        if prediction == 'Benign':
            report += """
- ✅ No significant abnormal features detected
- 🟢 Tissue appears normal
- 📅 Routine follow-up recommended
"""
        else:
            report += """
- ⚠️ Abnormal cellular structures detected
- 🔴 High attention in suspicious regions
- 🏥 **Specialist consultation recommended**
"""
        
        if prediction_id:
            report += f"\n---\n📋 **Record ID:** `{prediction_id}`"
        
        probs = {"Benign": benign_prob / 100, "Malignant": malignant_prob / 100}
        return probs, report
        
    except Exception as e:
        return None, f"❌ Error analyzing image: {str(e)}"


# Build Gradio Interface
def create_gradio_app() -> gr.Blocks:
    """Create and return the Gradio application."""
    
    with gr.Blocks() as app:
        
        gr.Markdown("""
        # 🔬 Breast Cancer Histopathology Analysis
        
        Upload histopathology images to detect **benign** or **malignant** tissue using Deep Learning.
        
        **[📜 View History & Details](/history)** | **[API Documentation](/api/docs)** | **Model:** EfficientNet-B0 with Coordinate Attention
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    type="pil",
                    label="Upload Histopathology Image",
                    height=350
                )
                analyze_btn = gr.Button(
                    "🔍 Analyze Image",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                output_label = gr.Label(
                    label="Classification",
                    num_top_classes=2
                )
                output_report = gr.Markdown(label="Analysis Report")
        
        analyze_btn.click(
            fn=analyze_image,
            inputs=[input_image],
            outputs=[output_label, output_report]
        )
        
        gr.Markdown("""
        ---
        ### ℹ️ About
        
        - **Model:** EfficientNet-B0 with Coordinate Attention mechanism
        - **Training Data:** Breast Cancer Histopathology dataset
        - **Classes:** Benign, Malignant
        
        ⚠️ **Disclaimer:** This tool is for research and educational purposes only. 
        Always consult qualified medical professionals for diagnosis.
        """)
    
    return app


# Create the app instance
gradio_app = create_gradio_app()
