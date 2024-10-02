from gradio_client import Client
from PIL import Image
from io import BytesIO
import base64

client = Client("http://127.0.0.1:7860")

p = "./img1.png"

print(client.view_api())
print(f'path = {p}')

output = client.predict(
    image=p,
    fn_index=0  # Без начального слэша
)
print(output)

# # Предположим, что output — это список из двух base64 строк
# if isinstance(output, (list, tuple)) and len(output) == 2:
#     processed_image_base64, origin_image_base64 = output
    
#     # Сохранение обработанного изображения
#     img_data = base64.b64decode(processed_image_base64.split(",")[1])
#     img = Image.open(BytesIO(img_data))
#     img.save("processed_image.png")
#     print("Изображение успешно обработано и сохранено как processed_image.png")
    
#     # Сохранение оригинального изображения
#     origin_img_data = base64.b64decode(origin_image_base64.split(",")[1])
#     origin_img = Image.open(BytesIO(origin_img_data))
#     origin_img.save("original_image.png")
#     print("Оригинальное изображение сохранено как original_image.png")
# else:
#     print("Неожиданный формат вывода:", output)
