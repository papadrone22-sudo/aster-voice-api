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


@spaces.GPU(duration=60)
def generate(text, reference_audio):
    if not (text or "").strip():
        raise gr.Error("Text kosong.")
    if not reference_audio:
        raise gr.Error("Reference voice belum ada.")

    model = get_model()
    wav = model.generate(
        text=text,
        reference_wav_path=reference_audio,
        cfg_value=2.0,
        inference_timesteps=4,
    )
    return (model.tts_model.sample_rate, wav)


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Text"),
        gr.Audio(type="filepath", label="Reference Voice"),
    ],
    outputs=gr.Audio(label="Output"),
    api_name="generate",
)

demo.queue(default_concurrency_limit=1).launch()
