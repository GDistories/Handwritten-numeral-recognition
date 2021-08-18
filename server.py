import json
import base64
import numpy as np
from io import BytesIO
import tensorflow as tf

from train import MODEL_PATH
from PIL import Image, ImageChops
from model import CNNModel
from flask import Flask, render_template, request
from datetime import timedelta

# Use flask server framework
app = Flask(__name__)
session = None
model = CNNModel()

# clear cache
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)


# Return to index page
@app.route('/')
def index():
    return render_template('show.html')


# Identify the img from the web page
@app.route('/classification', methods=['POST'])
def recognition():
    result = {"predict_digit": "error", "detect_img": "", "centering_img": "", "prob": {}}

    input_img = BytesIO(base64.urlsafe_b64decode(request.form['img']))
    predicted_value = predict(input_img)

    if predicted_value is not None:
        result["predict_digit"] = str(np.argmax(predicted_value))

        for i, data in enumerate(predicted_value):
            result["prob"][i] = float(data * 100)

    return json.dumps(result)


# Transform the input img shape to make it easier to be recognized by the model
def _centering(img):
    img_width, img_height = img.size[:2]
    left, top, right, bottom = img_width, img_height, -1, -1
    img_data = img.getdata()

    for y in range(img_height):
        yoffset = y * img_width
        for x in range(img_width):
            if img_data[yoffset + x] < 255:

                if x < left:
                    left = x
                if y < top:
                    top = y
                if x > right:
                    right = x
                if y > bottom:
                    bottom = y

    shiftX = (left + (right - left) // 2) - img_width // 2
    shiftY = (top + (bottom - top) // 2) - img_height // 2

    return ImageChops.offset(img, -shiftX, -shiftY)


# Forecast number
def predict(img_files):
    try:
        img = Image.open(img_files).convert('L')

    except IOError:
        print("Picture not found!")
        return None

    # Center input
    img = _centering(img)

    img.thumbnail((28, 28))  # resize to 28x28
    img = np.array(img, dtype=np.float32)
    img = 1 - np.array(img / 255)  # normalize
    img = img.reshape(1, 784)

    # Prediction
    res = session.run(model.softmax, feed_dict={model.input_shape: img, model.output: [[0.0] * 10], model.prob: 1.0})[0]
    return res


if __name__ == "__main__":

    if not tf.train.checkpoint_exists(MODEL_PATH):
        print("No model to load!")
        exit(1)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, MODEL_PATH)
        session = sess

        app.run(debug=True, host='0.0.0.0')
