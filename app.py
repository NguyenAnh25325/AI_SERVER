from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ["class1","class2","class3"]  # sửa lại

def preprocess(image):
    img = image.resize((224,224))
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image = Image.open(file).convert("RGB")

    img = preprocess(image)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    index = int(np.argmax(output))

    return jsonify({
        "result": class_names[index]
    })

@app.route('/')
def home():
    return "Server running!"

if __name__ == '__main__':
    app.run()