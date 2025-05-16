from flask import Flask, request, render_template, send_file
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from PIL import Image
import io

app = Flask(__name__)

# Detectar si hay GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar el modelo solo una vez
model = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
).to(device)

# Transformaciones
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        image = Image.open(file.stream).convert("RGB")
        processed_image = remove_background(image)
        
        # Convertimos la imagen a bytes
        img_io = io.BytesIO()
        processed_image.save(img_io, format="PNG")
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/png')
    
    return '''
        <h1>Background Remover</h1>
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="image">
            <input type="submit" value="Process">
        </form>
    '''

def remove_background(image):
    original_size = image.size
    input_tensor = transform_image(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid().cpu()
    
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(original_size)
    
    image.putalpha(mask)
    return image

if __name__ == "__main__":
    app.run(debug=True)
