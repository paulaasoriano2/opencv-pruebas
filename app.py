from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

html_colors = {
    "negro": (0,0,0),
    "blanco": (255,255,255),
    "rojo": (255,0,0),
    "verde": (0,128,0),
    "azul": (0,0,255),
    "amarillo": (255,255,0),
    "naranja": (255,165,0),
    "rosa": (255,192,203),
    "morado": (128,0,128),
    "cian": (0,255,255),
    "gris": (128,128,128),
    "marrón": (165,42,42),
    "lavanda": (230,230,250),
    "beige": (245,245,220),
    "dorado": (255,215,0),
    "plata": (192,192,192)
}

def closest_color_name(rgb):
    r, g, b = rgb
    min_dist = float('inf')
    name = "desconocido"
    for color_name, (cr, cg, cb) in html_colors.items():
        dist = np.sqrt((r-cr)**2 + (g-cg)**2 + (b-cb)**2)
        if dist < min_dist:
            min_dist = dist
            name = color_name
    return name

@app.route("/detect-color", methods=["POST"])
def detect_color():
    file = request.files["image"]

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_small = cv2.resize(img, (100,100))

    pixels = img_small.reshape(-1,3)
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)

    total_pixels = pixels.shape[0]

    color_info = {}
    for color, count in zip(unique_colors, counts):
        name = closest_color_name(color)
        color_info[name] = color_info.get(name,0) + count

    for name in color_info:
        color_info[name] = (color_info[name]/total_pixels)*100

    sorted_colors = sorted(color_info.items(), key=lambda x:x[1], reverse=True)

    dominant_name, dominant_percent = sorted_colors[0]

    return jsonify({
        "dominant_color": dominant_name,
        "percentage": dominant_percent,
        "all_colors": sorted_colors
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)