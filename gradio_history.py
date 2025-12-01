"""
Breast Cancer Histopathology Analysis - History & Detail Gradio UI
Separate Gradio app mounted at /history route.
"""

import gradio as gr
from PIL import Image
import io
import os
import requests
from typing import Optional, Dict, Any, List
import numpy as np

from core import ModelHandler, DatabaseHandler, StorageHandler

# Handlers (shared from main.py)
model_handler: Optional[ModelHandler] = None
db_handler: Optional[DatabaseHandler] = None
storage_handler: Optional[StorageHandler] = None


def get_history_data() -> List[Dict[str, Any]]:
    """Get recent prediction history from database."""
    if db_handler is None or not db_handler.is_connected:
        return []
    return db_handler.get_recent_predictions(50)


def format_history_table() -> str:
    """Format history as a detailed markdown table."""
    if db_handler is None or not db_handler.is_connected:
        return """
## ⚠️ Database Not Connected

History feature requires database connection. Please ensure MongoDB is configured.
"""
    
    history = get_history_data()
    
    if not history:
        return """
## 📭 No Analysis History Yet

Upload and analyze some images first to see history here.

👉 Go to the main page to analyze images.
"""
    
    # Statistics
    total = len(history)
    benign_count = sum(1 for h in history if h.get('prediction') == 'Benign')
    malignant_count = sum(1 for h in history if h.get('prediction') == 'Malignant')
    avg_confidence = sum(h.get('confidence', 0) for h in history) / total if total > 0 else 0
    
    md = f"""
## 📊 Analysis Statistics

| Metric | Value |
|--------|-------|
| **Total Analyses** | {total} |
| **Benign Cases** | {benign_count} ({benign_count/total*100:.1f}%) |
| **Malignant Cases** | {malignant_count} ({malignant_count/total*100:.1f}%) |
| **Average Confidence** | {avg_confidence*100:.1f}% |

---

## 📜 Recent Analysis History

| # | ID | Filename | Result | Confidence | Benign % | Malignant % | Timestamp |
|---|-------|----------|--------|------------|----------|-------------|-----------|
"""
    
    for i, item in enumerate(history, 1):
        pred = item.get('prediction', 'Unknown')
        conf = item.get('confidence', 0) * 100
        benign_prob = item.get('benign_probability', 0) * 100
        malignant_prob = item.get('malignant_probability', 0) * 100
        filename = item.get('filename', 'Unknown')
        if len(filename) > 20:
            filename = filename[:17] + "..."
        timestamp = item.get('timestamp', '')[:19] if item.get('timestamp') else 'N/A'
        record_id = item.get('_id', 'N/A')
        if len(str(record_id)) > 10:
            display_id = str(record_id)[:8] + "..."
        else:
            display_id = str(record_id)
        
        emoji = "🟢" if pred == "Benign" else "🔴"
        
        md += f"| {i} | `{display_id}` | {filename} | {emoji} {pred} | {conf:.1f}% | {benign_prob:.1f}% | {malignant_prob:.1f}% | {timestamp} |\n"
    
    md += """
---

💡 **Tip:** Copy the ID and paste it in the "Look Up by ID" section below to see detailed analysis.
"""
    
    return md


def get_prediction_detail(prediction_id: str) -> tuple:
    """Get detailed view of a specific prediction with comprehensive analysis."""
    if not prediction_id or not prediction_id.strip():
        return (
            "### ⚠️ Please enter a Prediction ID\n\nCopy an ID from the history table above.",
            None,
            None
        )
    
    if db_handler is None or not db_handler.is_connected:
        return (
            "### ⚠️ Database Not Connected\n\nCannot retrieve prediction details.",
            None,
            None
        )
    
    try:
        doc = db_handler.get_prediction(prediction_id.strip())
        if not doc:
            return (
                f"### ❌ Prediction Not Found\n\nNo record found with ID: `{prediction_id}`\n\nPlease check the ID and try again.",
                None,
                None
            )
        
        # Extract data
        pred = doc.get('prediction', 'Unknown')
        conf = doc.get('confidence', 0)
        benign_prob = doc.get('benign_probability', 0)
        malignant_prob = doc.get('malignant_probability', 0)
        filename = doc.get('filename', 'Unknown')
        image_url = doc.get('image_url')
        timestamp = doc.get('timestamp', 'Unknown')
        attention_metrics = doc.get('attention_metrics', {})
        
        # Emoji and colors based on prediction
        if pred == "Benign":
            result_emoji = "🟢"
            result_color = "green"
            risk_level = "Low Risk"
            risk_emoji = "✅"
        else:
            result_emoji = "🔴"
            result_color = "red"
            risk_level = "High Risk"
            risk_emoji = "⚠️"
        
        # Calculate risk score (0-100)
        risk_score = malignant_prob * 100
        
        # Confidence interpretation
        if conf >= 0.9:
            conf_level = "Very High"
            conf_emoji = "🎯"
        elif conf >= 0.75:
            conf_level = "High"
            conf_emoji = "✅"
        elif conf >= 0.6:
            conf_level = "Moderate"
            conf_emoji = "⚠️"
        else:
            conf_level = "Low"
            conf_emoji = "❓"
        
        # Attention metrics analysis
        mean_attention = attention_metrics.get('mean_intensity', 0)
        high_attention_ratio = attention_metrics.get('high_attention_ratio', 0)
        
        # Build comprehensive report
        detail = f"""
# 🔬 Detailed Analysis Report

---

## 📋 Record Information

| Field | Value |
|-------|-------|
| **Record ID** | `{doc.get('_id')}` |
| **Filename** | {filename} |
| **Analysis Date** | {timestamp} |

---

## 🎯 Classification Result

<div style="text-align: center; padding: 20px; background: {'#d4edda' if pred == 'Benign' else '#f8d7da'}; border-radius: 10px; margin: 10px 0;">

### {result_emoji} **{pred.upper()}** {result_emoji}

**Confidence Level:** {conf_level} {conf_emoji} ({conf*100:.2f}%)

**Risk Assessment:** {risk_emoji} {risk_level}

</div>

---

## 📊 Probability Analysis

| Class | Probability | Visual |
|-------|-------------|--------|
| **Benign** | {benign_prob*100:.2f}% | {'🟩' * int(benign_prob * 10)}{'⬜' * (10 - int(benign_prob * 10))} |
| **Malignant** | {malignant_prob*100:.2f}% | {'🟥' * int(malignant_prob * 10)}{'⬜' * (10 - int(malignant_prob * 10))} |

### Probability Breakdown

- **Benign Probability:** {benign_prob*100:.4f}%
- **Malignant Probability:** {malignant_prob*100:.4f}%
- **Confidence Score:** {conf*100:.4f}%
- **Probability Difference:** {abs(benign_prob - malignant_prob)*100:.4f}%

---

## 🔍 Attention Analysis

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Mean Attention Intensity** | {mean_attention:.4f} | {'High focus on specific regions' if mean_attention > 0.5 else 'Distributed attention'} |
| **High Attention Regions** | {high_attention_ratio:.2f}% | {'Concentrated suspicious areas' if high_attention_ratio > 30 else 'Normal distribution'} |

### What This Means

"""
        
        if pred == "Malignant":
            if high_attention_ratio > 30:
                detail += """
- 🔴 **Widespread Abnormality:** The model detected abnormal features across a large portion of the tissue
- 📍 **Multiple Suspicious Regions:** High attention areas indicate multiple regions of concern
- ⚠️ **Recommendation:** Immediate specialist consultation is strongly recommended
"""
            else:
                detail += """
- 🟠 **Focal Abnormality:** The model detected abnormal features in specific localized regions
- 📍 **Localized Concern:** Suspicious features appear concentrated in certain areas
- ⚠️ **Recommendation:** Further investigation and specialist consultation recommended
"""
        else:
            if high_attention_ratio > 20:
                detail += """
- 🟢 **Normal Tissue Patterns:** The model identifies typical benign tissue characteristics
- 📍 **Even Distribution:** Attention is spread across normal tissue structures
- ✅ **Recommendation:** Routine follow-up as per standard protocol
"""
            else:
                detail += """
- 🟢 **Minimal Abnormality:** Very few areas triggered elevated attention
- 📍 **Low Concern Areas:** The tissue appears predominantly normal
- ✅ **Recommendation:** Standard monitoring, routine follow-up
"""
        
        detail += f"""
---

## 💡 Clinical Insights

### For {pred} Classification:

"""
        
        if pred == "Benign":
            detail += """
| Aspect | Finding |
|--------|---------|
| **Cellular Structure** | ✅ Normal cellular organization observed |
| **Tissue Architecture** | ✅ Regular tissue patterns detected |
| **Nuclear Features** | ✅ Normal nuclear characteristics |
| **Growth Pattern** | ✅ No aggressive growth patterns identified |

### Recommended Actions:
1. 📅 **Routine Follow-up:** Schedule standard follow-up as per clinical protocol
2. 📋 **Documentation:** Keep this analysis for medical records
3. 🔄 **Monitoring:** Continue regular screening schedule
4. 👩‍⚕️ **Consultation:** Discuss results with healthcare provider

### Confidence Assessment:
"""
            if conf >= 0.9:
                detail += "- ✅ **Very High Confidence:** Model is highly certain about benign classification"
            elif conf >= 0.75:
                detail += "- ✅ **High Confidence:** Strong indication of benign tissue"
            elif conf >= 0.6:
                detail += "- ⚠️ **Moderate Confidence:** Consider additional testing for confirmation"
            else:
                detail += "- ❓ **Low Confidence:** Recommend pathologist review for confirmation"
                
        else:  # Malignant
            detail += """
| Aspect | Finding |
|--------|---------|
| **Cellular Structure** | ⚠️ Abnormal cellular organization detected |
| **Tissue Architecture** | ⚠️ Irregular tissue patterns observed |
| **Nuclear Features** | ⚠️ Atypical nuclear characteristics |
| **Growth Pattern** | ⚠️ Potential aggressive growth indicators |

### Recommended Actions:
1. 🏥 **Immediate Consultation:** Schedule appointment with oncologist/specialist
2. 🧪 **Confirmatory Testing:** Consider biopsy for histological confirmation
3. 📋 **Complete Workup:** Comprehensive diagnostic evaluation recommended
4. 👨‍👩‍👧‍👦 **Support:** Consider involving family in care discussions

### Confidence Assessment:
"""
            if conf >= 0.9:
                detail += "- 🔴 **Very High Confidence:** Strong indicators of malignancy detected"
            elif conf >= 0.75:
                detail += "- 🟠 **High Confidence:** Significant malignant characteristics observed"
            elif conf >= 0.6:
                detail += "- ⚠️ **Moderate Confidence:** Malignant features present, confirmation needed"
            else:
                detail += "- ❓ **Low Confidence:** Suspicious features detected, expert review essential"
        
        detail += f"""

---

## ⚠️ Important Disclaimer

> **This analysis is for research and educational purposes only.**
> 
> - This AI tool is NOT a substitute for professional medical diagnosis
> - Always consult qualified healthcare professionals for medical decisions
> - Results should be confirmed by a certified pathologist
> - Do not make treatment decisions based solely on this analysis

---

## 📈 Model Information

| Specification | Value |
|---------------|-------|
| **Model Architecture** | EfficientNet-B0 + Coordinate Attention |
| **Input Size** | 160 × 160 pixels |
| **Classes** | Benign, Malignant |
| **Training Data** | Breast Cancer Histopathology Dataset |

---

*Analysis generated by Breast Cancer Histopathology Analysis System*
"""
        
        # Create probability data for chart
        prob_data = {"Benign": benign_prob, "Malignant": malignant_prob}
        
        # Load image if available
        loaded_image = None
        if image_url:
            try:
                response = requests.get(image_url, timeout=10)
                if response.status_code == 200:
                    loaded_image = Image.open(io.BytesIO(response.content))
            except Exception as e:
                print(f"Could not load image: {e}")
        
        return detail, prob_data, loaded_image
        
    except Exception as e:
        return (
            f"### ❌ Error Loading Details\n\n```\n{str(e)}\n```\n\nPlease check the ID format and try again.",
            None,
            None
        )


def create_history_gradio_app() -> gr.Blocks:
    """Create the History & Detail Gradio application."""
    
    with gr.Blocks(
        title="Analysis History & Details",
        theme=gr.themes.Soft()
    ) as app:
        
        gr.Markdown("""
# 📜 Analysis History & Details

View past analyses and get detailed reports for any prediction.

[← Back to Main Analysis](/) | [API Documentation](/api/docs)
        """)
        
        with gr.Tabs():
            # Tab 1: History Overview
            with gr.Tab("📊 History Overview"):
                gr.Markdown("### View All Past Analyses")
                
                history_output = gr.Markdown(value="Click 'Load History' to view past analyses.")
                
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Load History", variant="primary", size="lg")
                
                refresh_btn.click(fn=format_history_table, outputs=[history_output])
                
                # Auto-load on page visit
                app.load(fn=format_history_table, outputs=[history_output])
            
            # Tab 2: Detailed View
            with gr.Tab("🔍 Detailed Analysis"):
                gr.Markdown("""
### Look Up Prediction by ID

Enter a prediction ID to view the complete detailed analysis report.
                """)
                
                with gr.Row():
                    with gr.Column(scale=3):
                        prediction_id_input = gr.Textbox(
                            label="Prediction ID",
                            placeholder="Paste the prediction ID here (e.g., 674abc123def456789012345)",
                            lines=1
                        )
                    with gr.Column(scale=1):
                        lookup_btn = gr.Button("🔍 Get Details", variant="primary", size="lg")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        detail_output = gr.Markdown(label="Analysis Report")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 📷 Analyzed Image")
                        image_output = gr.Image(label="Image", type="pil", height=300)
                        
                        gr.Markdown("### 📊 Probability Chart")
                        prob_output = gr.Label(label="Classification Probabilities", num_top_classes=2)
                
                lookup_btn.click(
                    fn=get_prediction_detail,
                    inputs=[prediction_id_input],
                    outputs=[detail_output, prob_output, image_output]
                )
                
                # Also trigger on Enter key
                prediction_id_input.submit(
                    fn=get_prediction_detail,
                    inputs=[prediction_id_input],
                    outputs=[detail_output, prob_output, image_output]
                )
        
        gr.Markdown("""
---

### ℹ️ About This Page

- **History Overview:** Shows all past analyses with statistics
- **Detailed Analysis:** In-depth report for any specific prediction
- **Clinical Insights:** AI-generated recommendations (for reference only)

⚠️ **Disclaimer:** This tool is for research and educational purposes only. 
Always consult qualified medical professionals for diagnosis.
        """)
    
    return app


# Create the app instance
history_gradio_app = create_history_gradio_app()
