from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import os

app = Flask(__name__)

# Load model
interpreter = tf.lite.Interpreter(model_path="plant_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Thay bằng tên lớp thật của model bạn
class_names = ["class1", "class2", "class3", ...]   # ← Sửa lại đầy đủ

def preprocess(image: Image.Image) -> np.ndarray:
    img = image.resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0     # Chuẩn hóa
    img = np.expand_dims(img, axis=0)
    return img


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    try:
        image = Image.open(file).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image"}), 400

    img = preprocess(image)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    index = int(np.argmax(output[0]))
    confidence = float(np.max(output[0]))

    return jsonify({
        "result": class_names[index],
        "confidence": round(confidence * 100, 2),
        "class_index": index
    })


@app.route('/')
def home():
    return "AI Plant Classification Server is running!"


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
