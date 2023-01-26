import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

model_id = "prompthero/openjourney"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

iface = gr.Interface(generate_image, gr.inputs.Textbox(label="Enter a prompt"), gr.outputs.Image())
iface.launch()
