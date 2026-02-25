from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np

app = FastAPI()

@app.post("/upload")
async def upload_image(image: UploadFile = File(...)):
    # Leer bytes de la foto
    contents = await image.read()

    # Convertir a imagen OpenCV
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Procesar la imagen: convertir a gris (ejemplo)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Guardar resultado
    cv2.imwrite("resultado.jpg", gray)

    return {"status": "ok", "message": "Imagen procesada"}