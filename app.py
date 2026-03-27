from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import os

app = Flask(__name__)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="plant_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Thứ tự class đúng theo alphabet của thư mục train
class_names = [
    "Tomato__Bacterial_spot",
    "Tomato__Early_blight",
    "Tomato__Late_blight",
    "Tomato__Leaf_Mold",
    "Tomato__Septoria_leaf_spot",
    "Tomato__Spider_mites_Two-spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato__healthy"
]

def preprocess(image: Image.Image) -> np.ndarray:
    img = image.resize((128, 128))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "Vui lòng gửi ảnh với key = 'image'"}), 400

    file = request.files['image']
    
    try:
        image = Image.open(file).convert("RGB")
    except Exception:
        return jsonify({"error": "File không phải ảnh hợp lệ"}), 400

    img = preprocess(image)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    index = int(np.argmax(output[0]))
    confidence = float(np.max(output[0]))

    result = class_names[index]
    display_name = result.replace("Tomato__", "").replace("_", " ").strip()

    status = "Khỏe mạnh" if "healthy" in result.lower() else "Có bệnh"

    return jsonify({
        "result": display_name,
        "status": status,
        "confidence": round(confidence * 100, 2),
        "class_index": index
    })


@app.route('/')
def home():
    return " Tomato Disease Detection Server is running successfully!"


# Chỉ chạy khi test local bằng lệnh: python app.py
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
