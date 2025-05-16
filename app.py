from flask import Flask, request, render_template, send_file
from models.background_removal import remove_background
from PIL import Image
import io

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        image = Image.open(file.stream).convert("RGB")
        processed_image = remove_background(image)

        img_io = io.BytesIO()
        processed_image.save(img_io, format="PNG")
        img_io.seek(0)
        return send_file(img_io, mimetype="image/png")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)