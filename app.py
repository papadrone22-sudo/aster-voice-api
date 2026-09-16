import os
import tempfile
import threading

try:
    import spaces

    @spaces.GPU(duration=1)
    def _zerogpu_startup_probe():
        return None
except ImportError:
    pass

import gradio as gr
import scipy.io.wavfile
from pocket_tts import TTSModel

MODEL_ID = "anak10thn/pocket-tts-indonesian"
DEFAULT_VOICE = "hf://kyutai/tts-voices/alba-mackenna/casual.wav"

_model = None
_default_voice_state = None
_model_lock = threading.Lock()


def get_model():
    global _model, _default_voice_state
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = TTSModel.load_model(MODEL_ID)
                _default_voice_state = _model.get_state_for_audio_prompt(DEFAULT_VOICE)
    return _model


def generate_speech(text, reference_audio=None):
    try:
        text = (text or "").strip()
        if not text:
            return None, "ERROR: Masukkan teks terlebih dahulu."
        if len(text) > 2000:
            return None, "ERROR: Maksimal 2000 karakter per generasi."

        model = get_model()
        if reference_audio:
            voice_state = model.get_state_for_audio_prompt(reference_audio)
            mode = "Voice Clone"
        else:
            global _default_voice_state
            if _default_voice_state is None:
                _default_voice_state = model.get_state_for_audio_prompt(DEFAULT_VOICE)
            voice_state = _default_voice_state
            mode = "Voice Over"

        audio = model.generate_audio(voice_state, text)
        audio_np = audio.detach().cpu().numpy()
        fd, out_path = tempfile.mkstemp(prefix="aster_tts_", suffix=".wav")
        os.close(fd)
        scipy.io.wavfile.write(out_path, model.sample_rate, audio_np)
        return out_path, f"Selesai • {mode} • {model.sample_rate} Hz"
    except Exception as exc:
        return None, f"ERROR: {type(exc).__name__}: {exc}"


with gr.Blocks(title="Aster Pocket TTS") as demo:
    gr.Markdown("# Aster Pocket TTS 🇮🇩\nPocket TTS Bahasa Indonesia + Voice Cloning")
    text = gr.Textbox(label="Teks", placeholder="Tulis teks Bahasa Indonesia...", lines=5, max_lines=12)
    reference = gr.Audio(label="Referensi suara (opsional untuk voice cloning)", sources=["upload", "microphone"], type="filepath")
    generate = gr.Button("Generate", variant="primary")
    output = gr.Audio(label="Hasil", type="filepath")
    status = gr.Textbox(label="Status", interactive=False)
    generate.click(fn=generate_speech, inputs=[text, reference], outputs=[output, status], api_name="generate")

demo.queue(default_concurrency_limit=1).launch()
