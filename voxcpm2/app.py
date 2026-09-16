import spaces
import re
import threading
import gradio as gr
from voxcpm import VoxCPM

MODEL_ID = "openbmb/VoxCPM2"
_model = None
_model_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = VoxCPM.from_pretrained(
                    MODEL_ID,
                    load_denoiser=False,
                    device="cuda",
                    optimize=False,
                )
    return _model


def _clean_control(control: str) -> str:
    return re.sub(r"[()（）]", "", (control or "")).strip()


@spaces.GPU(duration=120)
def generate(
    text,
    reference_audio,
    control_instruction,
    ultimate_cloning,
    reference_transcript,
    cfg_value,
    inference_timesteps,
    seed,
):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Text kosong.")

    control = _clean_control(control_instruction)
    transcript = (reference_transcript or "").strip()
    seed_value = int(seed) if seed is not None else None
    model = get_model()

    kwargs = {
        "text": text,
        "cfg_value": float(cfg_value),
        "inference_timesteps": int(inference_timesteps),
        "seed": seed_value,
    }

    if ultimate_cloning:
        if not reference_audio:
            raise gr.Error("Ultimate Cloning membutuhkan reference voice.")
        if not transcript:
            raise gr.Error("Ultimate Cloning membutuhkan transcript persis dari reference voice.")
        kwargs.update(
            prompt_wav_path=reference_audio,
            prompt_text=transcript,
            reference_wav_path=reference_audio,
        )
        mode = "Ultimate Cloning"
    else:
        if control:
            kwargs["text"] = f"({control}){text}"
        if reference_audio:
            kwargs["reference_wav_path"] = reference_audio
            mode = "Controllable Cloning" if control else "Voice Cloning"
        else:
            mode = "Voice Design" if control else "TTS"

    wav = model.generate(**kwargs)
    sample_rate = model.tts_model.sample_rate
    status = (
        f"Selesai • {mode} • {sample_rate} Hz • "
        f"{int(inference_timesteps)} steps • seed {seed_value}"
    )
    return (sample_rate, wav), status


with gr.Blocks(title="Aster VoxCPM2") as demo:
    gr.Markdown(
        "# Aster VoxCPM2 🎙️\n"
        "Voice Design + Controllable Voice Cloning + Ultimate Cloning"
    )

    text = gr.Textbox(
        label="Text",
        placeholder="Tulis teks Bahasa Indonesia...",
        lines=5,
    )
    reference_audio = gr.Audio(
        label="Reference Voice",
        sources=["upload", "microphone"],
        type="filepath",
    )
    control_instruction = gr.Textbox(
        label="Control Instruction",
        placeholder="Contoh: warm, intimate, slightly husky, calm, slow pace",
        lines=2,
        info="Atur emosi, pace, ekspresi, tone, dan style. Diabaikan saat Ultimate Cloning aktif.",
    )

    with gr.Accordion("Ultimate Cloning", open=False):
        ultimate_cloning = gr.Checkbox(
            label="Aktifkan Ultimate Cloning",
            value=False,
        )
        reference_transcript = gr.Textbox(
            label="Transcript Reference Voice",
            placeholder="Tulis transcript persis dari audio reference...",
            lines=3,
        )

    with gr.Accordion("Advanced", open=False):
        cfg_value = gr.Slider(
            minimum=1.0,
            maximum=3.0,
            value=2.0,
            step=0.1,
            label="CFG Guidance",
        )
        inference_timesteps = gr.Slider(
            minimum=4,
            maximum=20,
            value=10,
            step=1,
            label="Inference Steps",
        )
        seed = gr.Number(
            value=42,
            precision=0,
            label="Seed",
        )

    generate_button = gr.Button("Generate", variant="primary")
    output = gr.Audio(label="Output")
    status = gr.Textbox(label="Status", interactive=False)

    generate_button.click(
        fn=generate,
        inputs=[
            text,
            reference_audio,
            control_instruction,
            ultimate_cloning,
            reference_transcript,
            cfg_value,
            inference_timesteps,
            seed,
        ],
        outputs=[output, status],
        api_name="generate",
    )


demo.queue(default_concurrency_limit=1).launch()
