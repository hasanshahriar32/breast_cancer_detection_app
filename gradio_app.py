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


def get_history() -> str:
    """Get recent prediction history."""
    if db_handler is None or not db_handler.is_connected:
        return "⚠️ Database not connected. History unavailable."
    
    try:
        recent = db_handler.get_recent_predictions(10)
        if not recent:
            return "📭 No analysis history yet."
        
        history_md = "## 📜 Recent Analysis History\n\n"
        history_md += "| # | File | Result | Confidence | Time |\n"
        history_md += "|---|------|--------|------------|------|\n"
        
        for i, item in enumerate(recent, 1):
            pred = item.get('prediction', 'Unknown')
            conf = item.get('confidence', 0) * 100
            filename = item.get('filename', 'Unknown')[:20]
            timestamp = item.get('timestamp', '')[:19]
            emoji = "🟢" if pred == "Benign" else "🔴"
            history_md += f"| {i} | {filename} | {emoji} {pred} | {conf:.1f}% | {timestamp} |\n"
        
        return history_md
        
    except Exception as e:
        return f"❌ Error loading history: {str(e)}"


def get_prediction_detail(prediction_id: str) -> str:
    """Get detailed view of a specific prediction."""
    if not prediction_id:
        return "Enter a prediction ID to view details."
    
    if db_handler is None or not db_handler.is_connected:
        return "⚠️ Database not connected."
    
    try:
        doc = db_handler.get_prediction(prediction_id.strip())
        if not doc:
            return f"❌ Prediction not found: `{prediction_id}`"
        
        pred = doc.get('prediction', 'Unknown')
        conf = doc.get('confidence', 0) * 100
        emoji = "🟢" if pred == "Benign" else "🔴"
        
        detail = f"""
## 📋 Prediction Details

**ID:** `{doc.get('_id')}`

**File:** {doc.get('filename', 'Unknown')}

**Result:** {emoji} **{pred}** ({conf:.1f}% confidence)

| Metric | Value |
|--------|-------|
| Benign Probability | {doc.get('benign_probability', 0) * 100:.1f}% |
| Malignant Probability | {doc.get('malignant_probability', 0) * 100:.1f}% |

**Timestamp:** {doc.get('timestamp', 'Unknown')}
"""
        
        if doc.get('image_url'):
            detail += f"\n**Image URL:** [{doc.get('image_url')}]({doc.get('image_url')})"
        
        return detail
        
    except Exception as e:
        return f"❌ Error: {str(e)}"


# Build Gradio Interface
def create_gradio_app() -> gr.Blocks:
    """Create and return the Gradio application."""
    
    with gr.Blocks() as app:
        
        gr.Markdown("""
        # 🔬 Breast Cancer Histopathology Analysis
        
        Upload histopathology images to detect **benign** or **malignant** tissue using Deep Learning.
        
        **[📜 View History & Details](/history)** | **[API Documentation](/api/docs)** | **Model:** EfficientNet-B0 with Coordinate Attention
        """)
        
        with gr.Tabs():
            # Tab 1: Analysis
            with gr.Tab("🔍 Analyze"):
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
            
            # Tab 2: History
            with gr.Tab("📜 History"):
                history_output = gr.Markdown()
                refresh_btn = gr.Button("🔄 Refresh History", variant="secondary")
                refresh_btn.click(fn=get_history, outputs=[history_output])
                
                gr.Markdown("---")
                
                gr.Markdown("### 🔎 Look Up Prediction")
                with gr.Row():
                    prediction_id_input = gr.Textbox(
                        label="Prediction ID",
                        placeholder="Enter prediction ID (e.g., 507f1f77bcf86cd799439011)"
                    )
                    lookup_btn = gr.Button("🔍 Look Up", variant="secondary")
                
                detail_output = gr.Markdown()
                lookup_btn.click(
                    fn=get_prediction_detail,
                    inputs=[prediction_id_input],
                    outputs=[detail_output]
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
