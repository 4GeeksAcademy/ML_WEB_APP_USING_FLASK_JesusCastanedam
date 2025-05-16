import torch
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
).to(device)

transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),  # <-- Forzamos a 1024x1024
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def remove_background(image: Image.Image) -> Image.Image:
    # ⚠️ Redimensionar si es muy grande
    max_size = 1024
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.ANTIALIAS)

    original_size = image.size
    input_tensor = transform_image(image).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid().cpu()

    pred = preds[0].squeeze()
    mask = transforms.ToPILImage()(pred).resize(original_size)
    image.putalpha(mask)
    return image
