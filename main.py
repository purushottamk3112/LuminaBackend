from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydub import AudioSegment
import whisper
import os
import tempfile

app = FastAPI()

# Load the Whisper model
model = whisper.load_model("small")  # Now using correct package

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Convert to WAV
        audio = AudioSegment.from_file(tmp_path)
        wav_path = tmp_path.replace(".webm", ".wav")
        audio.export(wav_path, format="wav")

        # Transcribe
        result = model.transcribe(wav_path)
        
        # Cleanup
        os.unlink(tmp_path)
        os.unlink(wav_path)

        return JSONResponse(content={"transcription": result["text"]})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
