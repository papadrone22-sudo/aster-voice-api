import os
import subprocess
import tempfile
import threading
from pathlib import Path

import gradio as gr
import scipy.io.wavfile
import spaces
from pocket_tts import TTSModel

MODEL_CONFIG = "hf://anak10thn/pocket-tts-indonesian/indonesian_6l.yaml@17257664e384561c957b02ac92edd1a24807f0e5"
DEFAULT_VOICE = "hf://kyutai/tts-voices/alba-mackenna/casual.wav"
PRIMARY_EOS = -4.0
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


def prepare_reference_audio(reference_audio):
    source = Path(reference_audio)
    if not source.exists():
        raise gr.Error("Audio referensi tidak ditemukan.")

    fd, prepared_path = tempfile.mkstemp(prefix="aster_ref_", suffix=".wav")
    os.close(fd)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-vn", "-ac", "1", "-ar", "24000",
        "-t", "30", "-c:a", "pcm_s16le", prepared_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        if os.path.exists(prepared_path):
            os.unlink(prepared_path)
        raise gr.Error("Audio referensi tidak bisa dibaca. Gunakan WAV, MP3, M4A, atau format audio umum.")

    sample_rate, samples = scipy.io.wavfile.read(prepared_path)
    duration = float(samples.shape[0]) / float(sample_rate)
    if duration < 1.5:
        os.unlink(prepared_path)
        raise gr.Error("Audio referensi terlalu pendek. Gunakan minimal 1,5 detik suara yang jelas.")

    return prepared_path, duration


def make_voice_state(model, voice_source, is_reference=False):
    return model.get_state_for_audio_prompt(
        Path(voice_source) if is_reference else voice_source,
        truncate=False if is_reference else True,
    )


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
    prepared_reference = None
    reference_duration = None

    if reference_audio:
        prepared_reference, reference_duration = prepare_reference_audio(reference_audio)

    try:
        with _generate_lock:
            voice_source = prepared_reference if prepared_reference else DEFAULT_VOICE
            voice_state = make_voice_state(model, voice_source, is_reference=bool(prepared_reference))
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
                    voice_state = make_voice_state(model, voice_source, is_reference=bool(prepared_reference))
                    retry_audio, retry_duration = generate_once(model, voice_state, text)
                    if retry_duration > duration:
                        audio_np, duration = retry_audio, retry_duration
                finally:
                    model.eos_threshold = original_eos
    finally:
        if prepared_reference and os.path.exists(prepared_reference):
            os.unlink(prepared_reference)

    fd, output_path = tempfile.mkstemp(prefix="aster_tts_", suffix=".wav")
    os.close(fd)
    scipy.io.wavfile.write(output_path, model.sample_rate, audio_np)

    suffix = " • retry EOS -4" if retried else ""
    ref_info = f" • ref {reference_duration:.1f} detik" if reference_duration else ""
    return output_path, f"Selesai • {mode} • {duration:.2f} detik{ref_info}{suffix}"


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
