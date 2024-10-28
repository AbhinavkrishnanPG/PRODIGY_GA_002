import gradio as gr
import torch
from diffusers import StableDiffusionPipeline

# Load model
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe.to("cpu")  # or "cpu" if you're using a CPU


# Define Gradio interface
def generate_image(prompt):
    images = pipe(prompt).images
    return images[0]


# Create Gradio UI
iface = gr.Interface(
    fn=generate_image,
    inputs="text",
    outputs="image",
    title="Stable Diffusion Generator",
    description="Enter a text prompt to generate an image",
)

# Launch the interface
iface.launch()
