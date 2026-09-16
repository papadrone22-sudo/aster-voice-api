import os
import tempfile
import threading

import gradio as gr
import scipy.io.wavfile
import spaces
from pocket_tts import TTSModel

MODEL_CONFIG = "hf://anak10thn/pocket-tts-indonesian/indonesian_6l.yaml@17257664e384561c957b02ac92edd1a24807f0e5"
DEFAULT_VOICE = "hf://kyutai/tts-voices/alba-mackenna/casual.wav"
PRIMARY_EOS = -6.0
FALLBACK_EOS = -4.0

_model = None
_model_lock = threading.Lock()
_generate_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = TTSModel.load_model(
                    config=MODEL_CONFIG,
                    eos_threshold=PRIMARY_EOS,
                )
    return _model


def make_voice_state(model, reference_audio=None):
    source = reference_audio if reference_audio else DEFAULT_VOICE
    return model.get_state_for_audio_prompt(source, truncate=True)


def generate_once(model, voice_state, text):
    audio = model.generate_audio(
        voice_state,
        text,
        frames_after_eos=2,
        copy_state=True,
    )
    audio_np = audio.detach().cpu().numpy().squeeze()
    duration = float(audio_np.shape[-1]) / float(model.sample_rate)
    return audio_np, duration


@spaces.GPU(duration=60)
def generate_speech(text, reference_audio=None):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Masukkan teks terlebih dahulu.")
    if len(text) > 2000:
        raise gr.Error("Maksimal 2000 karakter per generasi.")

    model = get_model()
    mode = "Voice Clone" if reference_audio else "Voice Over"

    with _generate_lock:
        # Always rebuild the voice state inside the ZeroGPU request.
        # Cached model-state tensors can become stale across ZeroGPU allocations.
        voice_state = make_voice_state(model, reference_audio)
        audio_np, duration = generate_once(model, voice_state, text)

        # The Indonesian model card reports occasional silent/near-silent generations
        # at EOS -6.0, while -4.0 had zero silent generations in its eval sweep.
        # Retry only when the output is clearly abnormal for non-trivial text.
        retried = False
        if len(text.split()) >= 3 and duration < 0.8:
            retried = True
            original_eos = model.eos_threshold
            try:
                model.eos_threshold = FALLBACK_EOS
                voice_state = make_voice_state(model, reference_audio)
                retry_audio, retry_duration = generate_once(model, voice_state, text)
                if retry_duration > duration:
                    audio_np, duration = retry_audio, retry_duration
            finally:
                model.eos_threshold = original_eos

    fd, output_path = tempfile.mkstemp(prefix="aster_tts_", suffix=".wav")
    os.close(fd)
    scipy.io.wavfile.write(output_path, model.sample_rate, audio_np)

    suffix = " • retry EOS -4" if retried else ""
    return output_path, f"Selesai • {mode} • {duration:.2f} detik{suffix}"


with gr.Blocks(title="Aster Pocket TTS") as demo:
    gr.Markdown("# Aster Pocket TTS 🇮🇩\nPocket TTS Bahasa Indonesia + Voice Cloning")
    text = gr.Textbox(
        label="Teks",
        placeholder="Tulis teks Bahasa Indonesia...",
        lines=5,
        max_lines=12,
    )
    reference = gr.Audio(
        label="Referensi suara (opsional untuk voice cloning)",
        sources=["upload", "microphone"],
        type="filepath",
    )
    generate = gr.Button("Generate", variant="primary")
    output = gr.Audio(label="Hasil", type="filepath")
    status = gr.Textbox(label="Status", interactive=False)
    generate.click(
        fn=generate_speech,
        inputs=[text, reference],
        outputs=[output, status],
        api_name="generate",
    )

demo.queue(default_concurrency_limit=1).launch()
