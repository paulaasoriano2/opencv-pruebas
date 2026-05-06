from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

# Rangos de colores en HSV
color_ranges = {
    "red": [(0, 100, 100), (10, 255, 255)],
    "orange": [(10, 100, 100), (25, 255, 255)],
    "yellow": [(25, 100, 100), (35, 255, 255)],
    "green": [(35, 50, 50), (85, 255, 255)],
    "cyan": [(85, 50, 50), (100, 255, 255)],
    "blue": [(100, 50, 50), (140, 255, 255)],
    "purple": [(140, 50, 50), (160, 255, 255)],
    "pink": [(160, 50, 50), (180, 255, 255)]
}

def preprocess_image(img):
    # Redimensionar para rendimiento
    img = cv2.resize(img, (100, 100))

    # Normalizar iluminación (LAB)
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    l = cv2.equalizeHist(l)
    img_lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)

    return img

@app.route("/detect-color", methods=["POST"])
def detect_color():
    file = request.files["image"]

    # Leer imagen
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Convertir BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Preprocesado
    img = preprocess_image(img)

    # Convertir a HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    total_pixels = hsv.shape[0] * hsv.shape[1]

    color_info = {}

    for color_name, (lower, upper) in color_ranges.items():
        lower = np.array(lower)
        upper = np.array(upper)

        mask = cv2.inRange(hsv, lower, upper)

        count = np.sum(mask > 0)

        if count > 0:
            percentage = (count / total_pixels) * 100
            color_info[color_name] = percentage

    # Manejo de colores neutros (blanco, negro, gris)
    h, s, v = cv2.split(hsv)

    # Negro
    black_pixels = np.sum(v < 50)
    # Blanco
    white_pixels = np.sum((s < 50) & (v > 200))
    # Gris
    gray_pixels = np.sum((s < 50) & (v >= 50) & (v <= 200))

    if black_pixels > 0:
        color_info["black"] = (black_pixels / total_pixels) * 100
    if white_pixels > 0:
        color_info["white"] = (white_pixels / total_pixels) * 100
    if gray_pixels > 0:
        color_info["grey"] = (gray_pixels / total_pixels) * 100

    # Ordenar resultados
    sorted_colors = sorted(color_info.items(), key=lambda x: x[1], reverse=True)

    if len(sorted_colors) == 0:
        return jsonify({
            "dominant_color": "desconocido",
            "percentage": 0,
            "all_colors": []
        })

    dominant_name, dominant_percent = sorted_colors[0]

    return jsonify({
        "dominant_color": dominant_name,
        "percentage": dominant_percent,
        "all_colors": sorted_colors
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)