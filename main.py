from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydub import AudioSegment
import whisper  # Correct import from official package
import os
import tempfile

app = FastAPI()

# Load model at startup
model = whisper.load_model("small")

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
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

        return {"transcription": result["text"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
