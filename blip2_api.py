from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import io
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from typing import Dict, Any

app = FastAPI(
    title="Civic Complaint BLIP-2 Visual Analyzer", 
    description="Endpoint for zero-shot VQA extraction of image severity and description."
)

processor = None
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

@app.on_event("startup")
def load_model():
    """
    Called intimately by FastAPI once upon server boot up.
    Loads the monstrous 2.7B parameter model directly to system GPU/CPU limits.
    """
    global processor, model
    print(f"Server Booting -> Initializing BLIP-2 strictly on [{device.upper()}]...")
    
    model_id = "Salesforce/blip2-opt-2.7b"
    processor = Blip2Processor.from_pretrained(model_id)
    
    # FP16 dramatically reduces VRAM limits (if CUDA is present)
    model_dtype = torch.float16 if device == "cuda" else torch.float32
    model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=model_dtype)
    model.to(device)
    
    print("[✓] Optimization & Model State loaded gracefully.")

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Receives an image payload POST limit. Extracts the byte stream, evaluates 
    VQA prompts internally against Salesforce's BLIP model, and dumps a JSON map.
    """
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="The BLIP-2 model matrix is unavailable or still booting.")
        
    try:
        # 1. Decode Byte Stream cleanly to Pillow format
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Tensor optimization extraction
        dtype = torch.float16 if device == "cuda" else torch.float32

        # 2. Extract Baseline Detailed Visual Description
        desc_prompt = "Question: Detailed describe the specific issue, damage, or hazard in this image. Answer:"
        desc_inputs = processor(images=image, text=desc_prompt, return_tensors="pt").to(device, dtype)
        
        generated_ids = model.generate(**desc_inputs, max_new_tokens=60)
        description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        # 3. Force Zero-Shot Classifier Category Bounds
        cat_prompt = "Question: Classify this specific civic failure using exactly one of these labels (pothole, garbage, broken_pipe, broken_streetlight, other). Answer:"
        cat_inputs = processor(images=image, text=cat_prompt, return_tensors="pt").to(device, dtype)
        
        cat_ids = model.generate(**cat_inputs, max_new_tokens=10)
        category = processor.batch_decode(cat_ids, skip_special_tokens=True)[0].strip().lower().replace(" ", "_")
        
        # 4. Force Zero-Shot Hazard Severity Limit Extractor
        sev_prompt = "Question: Analysing the prior risk factor of the hazard shown, classify its magnitude strictly as Low, Medium, or High. Answer:"
        sev_inputs = processor(images=image, text=sev_prompt, return_tensors="pt").to(device, dtype)
        
        sev_ids = model.generate(**sev_inputs, max_new_tokens=10)
        severity = processor.batch_decode(sev_ids, skip_special_tokens=True)[0].strip().capitalize()
        
        # Handle noisy inferences collapsing to 'High' strictly if it generates an alien word
        if severity not in ["Low", "Medium", "High"]:
            severity = "Unknown"

        return {
            "description": description,
            "category": category,
            "severity": severity
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image inference or IO pipeline physically crashed: {str(e)}")

if __name__ == "__main__":
    # Start ASGI Server limits tracking API connections
    # Use standard limits `python blip2_api.py` to trigger natively
    uvicorn.run("blip2_api:app", host="0.0.0.0", port=8000, reload=False)
