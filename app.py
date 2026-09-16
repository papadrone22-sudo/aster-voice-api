import spaces
import os
import tempfile
import threading

import gradio as gr
import scipy.io.wavfile
from pocket_tts import TTSModel

MODEL_CONFIG = "hf://anak10thn/pocket-tts-indonesian/indonesian_6l.yaml@17257664e384561c957b02ac92edd1a24807f0e5"
DEFAULT_VOICE = "hf://kyutai/tts-voices/alba-mackenna/casual.wav"

_model = None
_default_voice_state = None
_model_lock = threading.Lock()
_generate_lock = threading.Lock()


def get_model():
    global _model, _default_voice_state
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = TTSModel.load_model(
                    config=MODEL_CONFIG,
                    eos_threshold=-6.0,
                )
                _default_voice_state = _model.get_state_for_audio_prompt(DEFAULT_VOICE)
    return _model


@spaces.GPU(duration=60)
def generate_speech(text, reference_audio=None):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Masukkan teks terlebih dahulu.")
    if len(text) > 2000:
        raise gr.Error("Maksimal 2000 karakter per generasi.")

    model = get_model()
    if reference_audio:
        voice_state = model.get_state_for_audio_prompt(reference_audio, truncate=True)
        mode = "Voice Clone"
    else:
        global _default_voice_state
        voice_state = _default_voice_state
        mode = "Voice Over"

    with _generate_lock:
        audio = model.generate_audio(voice_state, text)

    audio_np = audio.detach().cpu().numpy().squeeze()
    fd, output_path = tempfile.mkstemp(prefix="aster_tts_", suffix=".wav")
    os.close(fd)
    scipy.io.wavfile.write(output_path, model.sample_rate, audio_np)
    return output_path, f"Selesai • {mode} • {model.sample_rate} Hz"


with gr.Blocks(title="Aster Pocket TTS") as demo:
    gr.Markdown("# Aster Pocket TTS 🇮🇩\nPocket TTS Bahasa Indonesia + Voice Cloning")
    text = gr.Textbox(label="Teks", placeholder="Tulis teks Bahasa Indonesia...", lines=5, max_lines=12)
    reference = gr.Audio(label="Referensi suara (opsional untuk voice cloning)", sources=["upload", "microphone"], type="filepath")
    generate = gr.Button("Generate", variant="primary")
    output = gr.Audio(label="Hasil", type="filepath")
    status = gr.Textbox(label="Status", interactive=False)
    generate.click(fn=generate_speech, inputs=[text, reference], outputs=[output, status], api_name="generate")

demo.queue(default_concurrency_limit=1).launch()
