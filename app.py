import io
import os
import threading
import wave

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from pocket_tts import TTSModel

MODEL_CONFIG = "hf://anak10thn/pocket-tts-indonesian/indonesian_6l.yaml@17257664e384561c957b02ac92edd1a24807f0e5"
DEFAULT_VOICE = "hf://kyutai/tts-voices/alba-mackenna/casual.wav"

app = FastAPI(title="Aster Voice API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_model = None
_voice_state = None
_load_error = None
_model_lock = threading.Lock()


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)


def _ensure_model():
    global _model, _voice_state, _load_error
    if _model is not None:
        return _model, _voice_state

    with _model_lock:
        if _model is not None:
            return _model, _voice_state
        try:
            os.environ.setdefault("OMP_NUM_THREADS", "2")
            os.environ.setdefault("MKL_NUM_THREADS", "2")
            model = TTSModel.load_model(
                config=MODEL_CONFIG,
                quantize=True,
                eos_threshold=-6.0,
            )
            voice_state = model.get_state_for_audio_prompt(DEFAULT_VOICE)
            _model = model
            _voice_state = voice_state
            _load_error = None
            return _model, _voice_state
        except Exception as exc:
            _load_error = f"{type(exc).__name__}: {exc}"
            raise


def _to_wav_bytes(audio, sample_rate: int) -> bytes:
    samples = audio.detach().cpu().numpy()
    if np.issubdtype(samples.dtype, np.floating):
        samples = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    else:
        samples = samples.astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())
    return buffer.getvalue()


@app.get("/")
def root():
    return {
        "service": "aster-voice-api",
        "model": "Pocket TTS Indonesian 6L",
        "status": "ready-for-test",
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "model_loaded": _model is not None,
        "load_error": _load_error,
    }


@app.post("/tts")
def tts(request: TTSRequest):
    try:
        model, voice_state = _ensure_model()
        audio = model.generate_audio(voice_state, request.text)
        wav_bytes = _to_wav_bytes(audio, model.sample_rate)
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=aster-voice.wav"},
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Pocket TTS failed to initialize or generate audio: {type(exc).__name__}: {exc}",
        ) from exc
