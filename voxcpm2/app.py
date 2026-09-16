print("BOOT: start", flush=True)
import gradio as gr
print(f"BOOT: gradio {gr.__version__}", flush=True)
import spaces
print(f"BOOT: spaces {getattr(spaces, '__version__', 'unknown')}", flush=True)

@spaces.GPU(duration=10)
def ping(text):
    return text or "ok"

print("BOOT: decorator ready", flush=True)
demo = gr.Interface(fn=ping, inputs=gr.Textbox(label="Test"), outputs=gr.Textbox(label="Output"), api_name="ping")
print("BOOT: before launch", flush=True)
demo.launch()
