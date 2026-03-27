from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# =========================
# 1. LOAD TFLITE MODEL
# =========================
MODEL_PATH = "plant_model.tflite"
LABELS_PATH = "labels.txt"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]


# =========================
# 2. PREPROCESS IMAGE
# =========================
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))  # giống Colab

    img_array = np.array(img).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize

    return img_array


# =========================
# 3. PREDICT FUNCTION
# =========================
def predict(img_array):
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    result_idx = int(np.argmax(output_data[0]))
    confidence = float(output_data[0][result_idx]) * 100
    label = labels[result_idx]

    return result_idx, label, confidence


# =========================
# 4. API TEST ROUTE
# =========================
@app.route("/", methods=["GET"])
def home():
    return "AI Server is running 🚀"


# =========================
# 5. PREDICT API
# =========================
@app.route("/predict", methods=["POST"])
def predict_api():
    if "file" not in request.files:
        return jsonify({
            "error": "No file uploaded"
        }), 400

    file = request.files["file"]
    image_bytes = file.read()

    img_array = preprocess_image(image_bytes)
    idx, label, conf = predict(img_array)

    return jsonify({
        "class_index": idx,
        "result": label,
        "confidence": round(conf, 2),
        "status": "OK"
    })


# =========================
# 6. RUN SERVER (RENDER FIX PORT)
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
