from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference_sdk import InferenceHTTPClient
import tempfile
import os
from typing import Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="ASL Detection API", version="1.0.0")

# Add CORS middleware for meeting app integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Roboflow client
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
if not ROBOFLOW_API_KEY:
    raise ValueError("ROBOFLOW_API_KEY environment variable is required. Please set it in your .env file or environment.")

client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

@app.get("/")
async def root():
    return {"message": "ASL Detection API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/detect-asl")
async def detect_asl(image: UploadFile = File(...)) -> Dict[str, str]:
    """
    Detect ASL words from an uploaded image frame.
    Returns the highest confidence detection as text.
    """
    
    # Validate file type
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            content = await image.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Run inference using Roboflow
        result = client.infer(
            temp_file_path,
            model_id="specializationproject/asl-words-detection"
        )
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        # Extract predictions
        predictions = result.get('predictions', [])
        
        if not predictions:
            return {"detected_word": ""}
        
        # Find highest confidence prediction
        highest_confidence_prediction = max(predictions, key=lambda x: x['confidence'])
        detected_word = highest_confidence_prediction['class']
        
        return {"detected_word": detected_word}
        
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
