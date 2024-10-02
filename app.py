import os
import gradio as gr
from gradio_imageslider import ImageSlider
from loadimg import load_img
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from tqdm import tqdm
import time  # Если понадобится для демонстрации
from pathlib import Path
from datetime import datetime
OUTPUT = "outputs/"
torch.set_float32_matmul_precision(["high", "highest"][0])

birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
)
birefnet.to("cuda")
transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def fn(image):
    im = load_img(image, output_type="pil")
    im = im.convert("RGB")
    origin = im.copy()
    image = process(im)
    return (image, origin)


def process(image):
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to("cuda")
    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)
    return image



def process_files(files: list[str]):
    pathes = []
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    new_path = f"{OUTPUT}{current_time}"
    Path(new_path).mkdir(parents=True, exist_ok=True)

    # Обернём список файлов в tqdm для отображения прогресса
    for f in tqdm(files, desc="Обработка файлов", ncols=80):
        name_path = f.rsplit(".", 1)[0] + ".png"
        im = load_img(f, output_type="pil")
        im = im.convert("RGB")
        transparent = process(im)
        s = name_path.split("\\")[-1]
        name_path = f"{new_path}/{s}"
        transparent.save(name_path)
        pathes.append(name_path)
        # Если хотите отображать итерации в секунду, tqdm делает это автоматически
        # через параметр `rate` (скорость обработки)
        # Можно настроить отображение через параметр `bar_format`
    return pathes


slider1 = ImageSlider(label="birefnet", type="pil")
slider2 = ImageSlider(label="birefnet", type="pil")
image = gr.Image(label="Upload an image")
batch_input = gr.File(label="Upload an images", file_count="multiple", file_types=["image"])
png_files = gr.File(label="output png file", file_count="multiple")


chameleon = load_img("butterfly.jpg", output_type="pil")

slider = gr.Interface(
    fn, inputs=image, outputs=slider1, examples=[chameleon], api_name="image"
)

batch = gr.Interface(
    process_files,
    inputs=batch_input,
    outputs=png_files,
    examples=["butterfly.jpg"],
    api_name="batch",
)

app = gr.TabbedInterface([slider, batch], ["image", "batch"], title="background removal")

if __name__ == "__main__":
    app.launch(show_error=True, inbrowser=True)
