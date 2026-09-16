import threading
import gradio as gr
import spaces

MODEL_ID = "openbmb/VoxCPM2"
_model = None
_model_lock = threading.Lock()

def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from voxcpm import VoxCPM
                _model = VoxCPM.from_pretrained(
                    MODEL_ID,
                    load_denoiser=False,
                    device="cuda",
                    optimize=False,
                )
    return _model

@spaces.GPU(duration=120)
def generate(text, reference_audio, style_instruction, reference_transcript):
    text = (text or "").strip()
    style_instruction = (style_instruction or "").strip()
    reference_transcript = (reference_transcript or "").strip()
    if not text:
        raise gr.Error("Teks belum diisi.")
    if not reference_audio:
        raise gr.Error("Reference Voice belum di-upload.")

    model = get_model()
    kwargs = {
        "text": f"({style_instruction}){text}" if style_instruction else text,
        "reference_wav_path": reference_audio,
        "cfg_value": 2.0,
        "inference_timesteps": 10,
    }
    if reference_transcript:
        kwargs["prompt_wav_path"] = reference_audio
        kwargs["prompt_text"] = reference_transcript

    wav = model.generate(**kwargs)
    return (model.tts_model.sample_rate, wav)

with gr.Blocks(title="Aster VoxCPM2") as demo:
    gr.Markdown("# Aster VoxCPM2 🎙️\nVoice Cloning + Tone / Emotion / Style Control")
    text = gr.Textbox(label="Text", placeholder="Tulis teks Bahasa Indonesia...", lines=5)
    reference_audio = gr.Audio(label="Reference Voice", sources=["upload", "microphone"], type="filepath")
    style_instruction = gr.Textbox(
        label="Tone / Emotion / Style",
        placeholder="Contoh: warm, intimate, slightly husky, calm, slow pace",
        lines=2,
    )
    reference_transcript = gr.Textbox(
        label="Reference Transcript (opsional)",
        placeholder="Tulis transcript persis dari audio reference untuk cloning lebih kuat...",
        lines=3,
    )
    generate_button = gr.Button("Generate Voice", variant="primary")
    output = gr.Audio(label="Output Voice")
    generate_button.click(
        fn=generate,
        inputs=[text, reference_audio, style_instruction, reference_transcript],
        outputs=output,
        api_name="generate",
    )

demo.queue(default_concurrency_limit=1).launch()
