from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import os

app = Flask(__name__)

# Load model một lần khi khởi động app (tốt cho performance)
interpreter = tflite.Interpreter(model_path="plant_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# TODO: Thay bằng tên class thật của bạn (10-20 class về cây trồng)
class_names = [
    "class1", "class2", "class3", "class4", "class5",
    # ... thêm đầy đủ tên các lớp ở đây
]

def preprocess(image: Image.Image) -> np.ndarray:
    """Tiền xử lý ảnh cho model (224x224 là kích thước phổ biến)"""
    img = image.resize((224, 224))
    img = np.array(img, dtype=np.float32)
    
    # Chuẩn hóa nếu model của bạn cần (thường chia cho 255)
    img = img / 255.0
    
    img = np.expand_dims(img, axis=0)   # Thêm batch dimension
    return img


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    try:
        image = Image.open(file).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    img = preprocess(image)

    # Chạy inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    index = int(np.argmax(output[0]))        # Lấy class có xác suất cao nhất
    confidence = float(np.max(output[0]))    # Thêm độ tin cậy

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
