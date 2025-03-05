from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import whisper
import tempfile
import os
from pydub import AudioSegment
import uuid
from datetime import datetime
import shutil

app = FastAPI(
    title="LuminaText API",
    description="API for transcribing audio and video files",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Whisper model
model = whisper.load_model("small")

# Create temporary directory for file processing
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

def extract_audio(video_path: str) -> str:
    """Extract audio from video file."""
    audio_path = os.path.splitext(video_path)[0] + "_audio.wav"
    try:
        audio = AudioSegment.from_file(video_path)
        audio.export(audio_path, format="wav")
        return audio_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting audio: {str(e)}")

@app.post("/api/transcribe")
async def transcribe_file(file: UploadFile):
    """
    Transcribe audio or video file.
    Accepts: .mp3, .wav, .mp4
    Maximum file size: 25MB
    """
    # Validate file size (25MB limit)
    file_size = 0
    chunk_size = 1024 * 1024  # 1MB chunks
    
    # Create temporary file
    temp_file_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_{file.filename}")
    
    try:
        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                file_size += len(chunk)
                if file_size > 25 * 1024 * 1024:  # 25MB
                    raise HTTPException(
                        status_code=413,
                        detail="File size exceeds 25MB limit"
                    )
                buffer.write(chunk)
        
        # Process file based on type
        if file.filename.endswith(".mp4"):
            audio_path = extract_audio(temp_file_path)
            result = model.transcribe(audio_path)
            os.remove(audio_path)  # Clean up temporary audio file
        else:
            result = model.transcribe(temp_file_path)
        
        # Prepare response
        response_data = {
            "text": result["text"],
            "fileName": file.filename,
            "duration": "00:00",  # You can extract actual duration if needed
            "fileSize": f"{file_size / (1024 * 1024):.2f}MB",
            "date": datetime.now().isoformat()
        }
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)