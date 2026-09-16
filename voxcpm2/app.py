import gradio as gr
import spaces
from voxcpm import VoxCPM

model = VoxCPM.from_pretrained(
    "openbmb/VoxCPM2",
    load_denoiser=False,
    device="cuda",
    optimize=False,
)

@spaces.GPU(duration=10)
def generate(text, reference_audio):
    if not text.strip():
        raise gr.Error("Text kosong.")

    if not reference_audio:
        raise gr.Error("Reference voice belum ada.")

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

demo.launch()
