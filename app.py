from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

# Colores de referencia en RGB
reference_colors = {
    "red": np.array([255, 0, 0]),
    "orange": np.array([255, 165, 0]),
    "yellow": np.array([255, 255, 0]),
    "green": np.array([0, 255, 0]),
    "cyan": np.array([0, 255, 255]),
    "blue": np.array([0, 0, 255]),
    "purple": np.array([128, 0, 128]),
    "pink": np.array([255, 192, 203]),
    "white": np.array([255, 255, 255]),
    "black": np.array([0, 0, 0]),
    "grey": np.array([128, 128, 128])
}


def preprocess_image(img):

    # Redimensionar
    img = cv2.resize(img, (100, 100))

    # Crear versión en gris SOLO para el histograma
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Ecualizar histograma
    equalized = cv2.equalizeHist(gray)

    # Normalizar iluminación usando la imagen ecualizada
    normalized = cv2.normalize(
        equalized,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # Aplicar como máscara de iluminación
    normalized = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)

    # Mezclar con la original
    result = cv2.addWeighted(
        img,
        0.7,
        normalized,
        0.3,
        0
    )

    return result


def classify_color(pixel):

    min_distance = float("inf")
    detected_color = "unknown"

    for color_name, ref_color in reference_colors.items():

        # Distancia euclídea RGB
        distance = np.linalg.norm(pixel - ref_color)

        if distance < min_distance:
            min_distance = distance
            detected_color = color_name

    return detected_color


@app.route("/detect-color", methods=["POST"])
def detect_color():

    file = request.files["image"]

    # Leer imagen
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Preprocesado
    img = preprocess_image(img)

    total_pixels = img.shape[0] * img.shape[1]

    color_count = {}

    # Recorrer todos los píxeles
    for row in img:
        for pixel in row:

            color = classify_color(pixel)

            if color not in color_count:
                color_count[color] = 0

            color_count[color] += 1

    # Convertir a porcentaje
    color_info = {}

    for color, count in color_count.items():
        percentage = (count / total_pixels) * 100
        color_info[color] = percentage

    # Ordenar resultados
    sorted_colors = sorted(
        color_info.items(),
        key=lambda x: x[1],
        reverse=True
    )

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