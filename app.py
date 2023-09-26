import ipywidgets as widgets
from IPython.display import display
import torch
import numpy as np
import modin.pandas as pd
from PIL import Image
import pandas as pd
from diffusers import DiffusionPipeline
import gradio as gr

link0 = ''
link1 = 'DmatryMakeev/asic'
link2 = 'dreamlike-art/dreamlike-photoreal-2.0'
link3 = 'dreamlike-art/dreamlike-anime-1.0'
link4 = 'dreamlike-art/dreamlike-diffusion-1.0'
link5 = 'wavymulder/Analog-Diffusion'
link6 = 'coreco/seek.art_MEGA'
link7 = 'DucHaiten/DucHaitenAIart'
link8 = 'Protogen_x3.4_Official_Release'
link9 = 'Nacholmo/meinamixv7-diffusers'
dropdown = widgets.Dropdown(
    options=[('Выберети модель', link0),
             ('Photoreal-1.0', link1),
             ('Photoreal-2.0', link2),
             ('Anime-1.0', link3),
             ('Dreamlike-Art-diffusion', link4),
             ('Analog-Diffusion', link5),
             ('Seek.art_MEGA', link6),
             ('DucHaitenAIart', link7),
             ('Protogen_x3.4_Official_Release', link8),
             ('Nacholmo/meinamixv7-diffusers', link9)],
    description='Выберите ссылку модели Stable Diffusion:'
)

selected_link = None

def on_change(change):
    global selected_link
    if change['name'] == 'value' and change['new']:
        selected_link = dropdown.value
        print(f"Вы выбрали ссылку: {selected_link}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = DiffusionPipeline.from_pretrained(selected_link, torch_dtype=torch.float16, variant="fp16")
        pipe = pipe.to(device)

        def genie(Prompt, negative_prompt, height, width, scale, steps, seed):
            generator = torch.Generator(device=device).manual_seed(seed)
            image = pipe(Prompt, negative_prompt=negative_prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=scale, generator=generator).images[0]
            return image

        gr.Interface(
            fn=genie,
            inputs=[
                gr.inputs.Textbox(label='Что вы хотите, чтобы ИИ генерировал?'),
                gr.inputs.Textbox(label='Что вы не хотите, чтобы ИИ генерировал?', default='(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, wrinkles, old face'),
                gr.Slider(512, 1024, 768, step=128, label='Высота изображения'),
                gr.Slider(512, 1024, 768, step=128, label='Ширина изображения'),
                gr.Slider(1, maximum=15, value=10, step=0.1, label='Шкала расхождения'),
                gr.Slider(1, maximum=100, value=25, step=1, label='Количество итераций'),
                gr.Slider(label="Точка старта функции", minimum=1, step=1, maximum=9999999999999999, randomize=True)
            ],
            outputs='image',
            title='DIAMONIK7777 - txt2img - Multy - Model',
            description="<p style='text-align: center'>Будь в курсе обновлений <a href='https://vk.com/public221489796'>ПОДПИСАТЬСЯ</a></p>",
            article="<br><br><p style='text-align: center'>Генерация индивидуальной модели с собственной внешностью <a href='https://vk.com/im?sel=-221489796'>ПОДАТЬ ЗАЯВКУ</a></p><br><br><br><br><br>",
        ).launch(debug=True, max_threads=True, share=True, inbrowser=True)

display(dropdown)
dropdown.observe(on_change)
