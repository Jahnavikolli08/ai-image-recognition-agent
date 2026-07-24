from flask import Flask, render_template, request
import os
from flask import send_from_directory
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load AI model
model = MobileNetV2(weights="imagenet")

# Upload folder
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template("index.html")
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/predict", methods=["POST"])
def predict():

    image_file = request.files["image"]

    if image_file:

        # Save uploaded image
        file_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            image_file.filename
        )

        image_file.save(file_path)

        # Load image
        img = image.load_img(file_path, target_size=(224, 224))

        # Convert image to array
        img_array = image.img_to_array(img)

        # Expand dimensions
        img_array = np.expand_dims(img_array, axis=0)

        # Preprocess image
        img_array = preprocess_input(img_array)

        # Predict
        predictions = model.predict(img_array)

        # Decode prediction
        decoded = decode_predictions(predictions, top=1)[0][0]

        object_name = decoded[1].replace("_", " ").title()
        confidence = decoded[2] * 100

        return render_template(
           "result.html",
            object_name=object_name,
            confidence=f"{confidence:.2f}",
            image_path="/uploads/" + image_file.filename
        )

    return "No Image Uploaded!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)