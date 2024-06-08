from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import keras
import cv2

app = FastAPI()
model = keras.saving.load_model("ai/model.keras")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict_image_with_AI_model(file: UploadFile):
    saveImage(file)
    image = getModelImage(file)

    answerExpected = file.filename.replace(".png", "")
    print(f"Expected Answer: {answerExpected}")

    answerActual = getModelAnswer(image)
    print(f"Actual Answer: {answerActual}")

    answerEquals = int(answerExpected) == int(answerActual)
    print(f"Request answers equals: {answerEquals}")

    return {"equals": answerEquals }

def saveImage(file: UploadFile):
    file_location = f"files/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

def getModelImage(file: UploadFile):
    image = cv2.imread(f"files/{file.filename}")
    resized_image = cv2.resize(image, (64, 64))
    data = [resized_image]
    data = np.array(data, dtype="float") / 255.0
    return data

def getModelAnswer(image):
    predictions = model.predict(image)
    adjusted_predictions = np.argmax(predictions, axis=1)
    return adjusted_predictions[0]
